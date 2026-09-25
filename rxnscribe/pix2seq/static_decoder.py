"""
Greedy decoding for the Pix2Seq reaction transformer with a fixed batch, cached cross-attention keys/values,
a preallocated self-attention cache and the output grammar evaluated on the device.

The reference loop (Transformer.forward, eval branch) recomputes the key/value projections of the whole image memory
(~1.8k positions) in every decoder layer at every step, grows the self-attention cache with torch.cat, and runs the
tokenizer's grammar (a small state machine) in Python for every sample at every step, which forces a host sync.
Here the memory projections are computed once per image, the grammar is two lookup tables (state transition and
allowed-token mask) and the host checks for completion every `check_every` steps. On CUDA one decoding step can be
captured into a CUDA graph and replayed.

Semantics are those of the reference loop: grammar mask, then max over the whole vocabulary for the emitted token
and score, argmax over the first `num_vocal - 2` entries for the next input embedding, and the reference `end_lens`
bookkeeping (a sequence that never emits EOS keeps length 0).
"""
import math

import numpy as np
import torch
import torch.nn.functional as F


class GrammarTables:
    """The ReactionTokenizer state machine as tensors: state 0 is the initial (None) state."""

    def __init__(self, tokenizer, vocab_size):
        states = [None]
        transitions = []
        masks = []
        is_x = [bool(tokenizer.is_x(i)) for i in range(vocab_size)]
        x_token = is_x.index(True)
        non_x_token = tokenizer.EOS_ID
        queue = [None]
        index = {None: 0}
        while queue:
            state = queue.pop(0)
            row = []
            for token in (non_x_token, x_token):
                nxt = tokenizer.update_state(state, token)
                if nxt not in index:
                    index[nxt] = len(states)
                    states.append(nxt)
                    queue.append(nxt)
                row.append(index[nxt])
            transitions.append((index[state], row))
        table = [[0, 0] for _ in states]
        for s, row in transitions:
            table[s] = row
        for state in states:
            masks.append(np.asarray(tokenizer.output_mask(state), dtype=bool) if state is not None
                         else np.ones(vocab_size, dtype=bool))
        self.transition = torch.tensor(table, dtype=torch.long)          # [num_states, 2] (is_x = 0/1)
        self.mask = torch.from_numpy(np.stack(masks))                     # [num_states, vocab]
        self.is_x = torch.tensor(is_x, dtype=torch.long)                  # [vocab]
        self.states = states

    def to(self, device):
        self.transition = self.transition.to(device)
        self.mask = self.mask.to(device)
        self.is_x = self.is_x.to(device)
        return self


class StaticPix2SeqDecoder:

    def __init__(self, transformer, max_len, check_every=8, use_cuda_graph=False):
        self.t = transformer
        self.max_len = max_len
        self.check_every = check_every
        self.use_cuda_graph = use_cuda_graph
        self.layers = list(transformer.decoder.layers)
        self.num_layers = len(self.layers)
        self.heads = self.layers[0].self_attn.num_heads
        self.dim = transformer.d_model
        self.head_dim = self.dim // self.heads
        self.vocab = transformer.num_vocal
        self.eos = transformer.tokenizer.EOS_ID
        self.grammar = GrammarTables(transformer.tokenizer, self.vocab) \
            if transformer.tokenizer.output_constraint else None
        self._grammar_device = None
        self._graphs = {}

    @staticmethod
    def supported(transformer):
        return not transformer.decoder.layers[0].normalize_before and transformer.pred_eos

    # ---- building blocks -------------------------------------------------------------------------------------------

    def _memory_kv(self, memory, pos):
        """Cross-attention keys/values of every layer, in the [B*H, S, hd] layout of F.multi_head_attention_forward."""
        s, b, _ = memory.shape
        key_in = memory + pos
        kv = []
        for layer in self.layers:
            mha = layer.multihead_attn
            w_k, w_v = mha.in_proj_weight[self.dim:2 * self.dim], mha.in_proj_weight[2 * self.dim:]
            b_k, b_v = mha.in_proj_bias[self.dim:2 * self.dim], mha.in_proj_bias[2 * self.dim:]
            k = F.linear(key_in, w_k, b_k).view(s, b * self.heads, self.head_dim).transpose(0, 1)
            v = F.linear(memory, w_v, b_v).view(s, b * self.heads, self.head_dim).transpose(0, 1)
            kv.append((k, v))
        return kv

    def _cross_attend(self, mha, query, k, v):
        _, b, _ = query.shape
        q = F.linear(query, mha.in_proj_weight[:self.dim], mha.in_proj_bias[:self.dim])
        q = q.view(1, b * self.heads, self.head_dim).transpose(0, 1)
        q = q * math.sqrt(1.0 / float(self.head_dim))
        attn = torch.softmax(torch.bmm(q, k.transpose(-2, -1)), dim=-1)
        out = torch.bmm(attn, v).transpose(0, 1).contiguous().view(b, self.dim)
        return F.linear(out, mha.out_proj.weight, mha.out_proj.bias).view(1, b, self.dim)

    def _self_attend(self, attn_module, x, step, cache_k, cache_v, self_mask):
        n, b, c = x.shape
        qkv = attn_module.qkv(x).reshape(n, b, 3, self.heads, c // self.heads).permute(2, 1, 3, 0, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if isinstance(step, int):
            cache_k[:, :, step:step + 1] = k
            cache_v[:, :, step:step + 1] = v
            keys, values = cache_k[:, :, :step + 1], cache_v[:, :, :step + 1]
        else:
            cache_k.index_copy_(2, step, k)
            cache_v.index_copy_(2, step, v)
            keys, values = cache_k, cache_v
        attn = (q @ keys.transpose(-2, -1)) * attn_module.scale
        if self_mask is not None:
            attn = attn.masked_fill(self_mask, float('-inf'))
        attn = attn.softmax(dim=-1)
        out = (attn @ values).permute(2, 0, 1, 3).reshape(n, b, c)
        return attn_module.proj(out)

    def _step(self, input_embed, step, memory_kv, caches, self_mask=None):
        tgt = input_embed
        for i, layer in enumerate(self.layers):
            cache_k, cache_v = caches[i]
            tgt = layer.norm1(tgt + self._self_attend(layer.self_attn, tgt, step, cache_k, cache_v, self_mask))
            k, v = memory_kv[i]
            tgt = layer.norm2(tgt + self._cross_attend(layer.multihead_attn, tgt, k, v))
            tgt = layer.norm3(tgt + layer.linear2(layer.activation(layer.linear1(tgt))))
        hs = self.t.decoder.norm(tgt)
        logits = self.t.vocal_classifier(hs.transpose(0, 1))  # [B, 1, V]
        return F.log_softmax(logits, dim=-1)

    def _select(self, log_probs, state, prev_token, first_step):
        """Grammar mask, emitted token/score, next input embedding. Mirrors Transformer.forward."""
        if self.grammar is not None:
            g = self.grammar
            if isinstance(first_step, bool):
                state = g.transition[0, 0].expand_as(state) if first_step else \
                    g.transition[state, g.is_x[prev_token]]
            else:
                state = torch.where(first_step, g.transition[0, 0].expand_as(state),
                                    g.transition[state, g.is_x[prev_token]])
            log_probs = log_probs.masked_fill(g.mask[state].unsqueeze(1), -10000)
        score, pred_token = log_probs.max(dim=-1)                               # [B, 1]
        token = log_probs[:, :, :self.vocab - 2].argmax(dim=-1)                 # [B, 1]
        next_embed = self.t.vocal_embed(token.transpose(0, 1))                  # [1, B, D]
        return state, score, pred_token, next_embed

    def _ensure_grammar(self, device):
        if self.grammar is not None and self._grammar_device != device:
            self.grammar.to(device)
            self._grammar_device = device

    def _alloc_caches(self, b, dtype, device):
        shape = (b, self.heads, self.max_len, self.head_dim)
        return [(torch.zeros(shape, dtype=dtype, device=device), torch.zeros(shape, dtype=dtype, device=device))
                for _ in range(self.num_layers)]

    # ---- eager -----------------------------------------------------------------------------------------------------

    def _decode_eager(self, memory, pos):
        b, device = memory.shape[1], memory.device
        memory_kv = self._memory_kv(memory, pos)
        caches = self._alloc_caches(b, memory_kv[0][0].dtype, device)
        input_embed = self.t.det_embed.weight.unsqueeze(0).repeat(b, 1, 1).transpose(0, 1)
        state = torch.zeros(b, dtype=torch.long, device=device)
        prev_token = torch.zeros(b, dtype=torch.long, device=device)
        end = torch.zeros(b, dtype=torch.bool, device=device)
        end_lens = torch.zeros(b, dtype=torch.long, device=device)
        seqs, scores = [], []
        for step in range(self.max_len):
            log_probs = self._step(input_embed, step, memory_kv, caches)
            state, score, pred_token, input_embed = self._select(log_probs, state, prev_token, step == 0)
            seqs.append(pred_token)
            scores.append(score)
            prev_token = pred_token.squeeze(1)
            stop = prev_token.eq(self.eos)
            end_lens += step * (~end & stop)
            end = end | stop
            if step > 4 and (step + 1) % self.check_every == 0 and bool(end.all()):
                break
        return torch.cat(seqs, dim=1), torch.cat(scores, dim=1), end_lens

    # ---- CUDA graph ------------------------------------------------------------------------------------------------

    def _build_graph(self, b, dtype, device, mem_len):
        g = {}
        g['memory_kv'] = [(torch.zeros((b * self.heads, mem_len, self.head_dim), dtype=dtype, device=device),
                           torch.zeros((b * self.heads, mem_len, self.head_dim), dtype=dtype, device=device))
                          for _ in range(self.num_layers)]
        g['caches'] = self._alloc_caches(b, dtype, device)
        g['input_embed'] = torch.zeros((1, b, self.dim), dtype=torch.float, device=device)
        g['state'] = torch.zeros(b, dtype=torch.long, device=device)
        g['prev_token'] = torch.zeros(b, dtype=torch.long, device=device)
        g['end'] = torch.zeros(b, dtype=torch.bool, device=device)
        g['end_lens'] = torch.zeros(b, dtype=torch.long, device=device)
        g['step'] = torch.zeros(1, dtype=torch.long, device=device)
        g['positions'] = torch.arange(self.max_len, device=device)
        g['seqs'] = torch.zeros((b, self.max_len), dtype=torch.long, device=device)
        g['scores'] = torch.zeros((b, self.max_len), dtype=torch.float, device=device)

        def body():
            step = g['step']
            self_mask = (g['positions'] > step).view(1, 1, 1, -1)
            log_probs = self._step(g['input_embed'], step, g['memory_kv'], g['caches'], self_mask)
            state, score, pred_token, next_embed = self._select(log_probs, g['state'], g['prev_token'], step.eq(0))
            g['state'].copy_(state)
            g['seqs'].index_copy_(1, step, pred_token)
            g['scores'].index_copy_(1, step, score.float())
            tok = pred_token.squeeze(1)
            g['prev_token'].copy_(tok)
            stop = tok.eq(self.eos)
            g['end_lens'].add_(step * (~g['end'] & stop))
            g['end'].copy_(g['end'] | stop)
            g['input_embed'].copy_(next_embed)
            g['step'].add_(1)

        stream = torch.cuda.Stream(device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                body()
        torch.cuda.current_stream(device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            body()
        g['graph'] = graph
        return g

    def _decode_graph(self, memory, pos):
        b, device = memory.shape[1], memory.device
        memory_kv = self._memory_kv(memory, pos)
        dtype = memory_kv[0][0].dtype
        key = (b, dtype, memory.shape[0], torch.backends.cuda.matmul.allow_tf32)
        if key not in self._graphs:
            self._graphs[key] = self._build_graph(b, dtype, device, memory.shape[0])
        g = self._graphs[key]
        for (dk, dv), (sk, sv) in zip(g['memory_kv'], memory_kv):
            dk.copy_(sk)
            dv.copy_(sv)
        g['input_embed'].copy_(self.t.det_embed.weight.unsqueeze(0).repeat(b, 1, 1).transpose(0, 1))
        for name in ('state', 'prev_token', 'end', 'end_lens', 'step'):
            g[name].zero_()
        steps = self.max_len
        for step in range(self.max_len):
            g['graph'].replay()
            if step > 4 and (step + 1) % self.check_every == 0 and bool(g['end'].all()):
                steps = step + 1
                break
        return g['seqs'][:, :steps], g['scores'][:, :steps], g['end_lens']

    # ---- public ----------------------------------------------------------------------------------------------------

    @torch.no_grad()
    def __call__(self, memory, pos):
        """memory, pos: [S, B, D]. Returns (pred_seq, pred_scores) lists like Transformer.forward."""
        self._ensure_grammar(memory.device)
        use_graph = self.use_cuda_graph and memory.is_cuda
        seqs, scores, end_lens = (self._decode_graph if use_graph else self._decode_eager)(memory, pos)
        seqs, scores, end_lens = seqs.cpu(), scores.cpu(), end_lens.cpu()
        return [s[:e] for e, s in zip(end_lens, seqs)], [s[:e] for e, s in zip(end_lens, scores)]
