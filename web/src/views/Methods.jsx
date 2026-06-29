import React from 'react'

export default function Methods() {
  return (
    <div className="page">
      <h1>Methods</h1>

      <section className="section" aria-labelledby="data">
        <h2 id="data">Data and Splits</h2>
        <p>
          PrimeKG (Chandak et al., 2023), pulled from Harvard Dataverse (
          <a href="https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXA7BM"
             target="_blank" rel="noopener noreferrer">doi:10.7910/DVN/IXA7BM</a>
          ). 8,100,498 edges across 129,375 nodes. License: CC0.
        </p>
        <table>
          <thead>
            <tr><th>Split</th><th>Description</th><th>Test diseases</th></tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>Standard</strong></td>
              <td>Random edge hold-out; a disease may appear in train and test</td>
              <td>Diseases present in training</td>
            </tr>
            <tr>
              <td><strong>Zero-shot</strong></td>
              <td>Held-out diseases keep zero treatment edges in train</td>
              <td>641 diseases, all drug-naive in train</td>
            </tr>
          </tbody>
        </table>
        <p>
          Two relations are excluded from message passing for VRAM reasons —{' '}
          <code>anatomy_protein_present</code> (3.03M) and <code>drug_drug</code> (2.67M) — leaving
          ~2.4M of 8.1M edges. Recorded as a scaled-reproduction deviation.
        </p>
      </section>

      <section className="section" aria-labelledby="models">
        <h2 id="models">Models</h2>

        <h3>Plain GNN (Q1 KG condition)</h3>
        <p>
          Two-layer HGT encoder, learnable <code>nn.Embedding</code> per node type (Xavier),
          hidden dim 64, dot-product scoring. Phase 1 link-prediction pretrain (30 epochs), Phase 2
          therapeutic fine-tune (≤100 epochs, patience 10).
        </p>

        <h3>Plain GNN, no-KG (Q1 control)</h3>
        <p>
          Identical but <code>depth=0</code> — embeddings with no message passing. Both share the
          Phase 1 pretrain, so “no-KG” means no aggregation, not no KG-informed initialization.
        </p>

        <h3>Scaled TxGNN</h3>
        <div className="finding-box">
          <p>
            <strong>Architecture note:</strong> TxGNN’s encoder is a Heterogeneous Graph Transformer
            (HGT). “Transformer” names its multi-head attention aggregation — it is a{' '}
            <strong>graph neural network</strong>, not a language model. No LLM is trained here.
          </p>
        </div>
        <p>
          The HGT encoder plus a <code>DiseaseAffinityHead</code> (cosine projection, k=5 support
          diseases). Two-phase training: link-prediction pretrain, then therapeutic task plus a
          triplet metric loss (<code>sim_loss_weight=0.3</code>).
        </p>
        <div className="warn-box">
          <p>
            <strong>Deviations from Huang et al. (2024):</strong> hidden_dim 512→64, layers 3→2,
            heads 8→4, learnable embeddings instead of pre-trained features. Everything is tagged{' '}
            <code>scaled_reproduction</code>.
          </p>
        </div>

        <h3>Alternatives (Q2)</h3>
        <ul>
          <li><strong>JointTaskNet</strong> — KG link-prediction + therapeutic loss from epoch 1.</li>
          <li><strong>ContrastiveNet</strong> — InfoNCE disease similarity + therapeutic loss jointly.</li>
        </ul>

        <h3>Ablations (Q6)</h3>
        <ul>
          <li><strong>attn=ON</strong> — full HGT attention.</li>
          <li><strong>attn=OFF</strong> — SAGEConv mean aggregation instead.</li>
        </ul>
      </section>

      <section className="section" aria-labelledby="eval">
        <h2 id="eval">Evaluation</h2>
        <table>
          <tbody>
            <tr><th scope="row">Primary metric</th><td>AUPRC</td></tr>
            <tr><th scope="row">Secondary metric</th><td>AUROC</td></tr>
            <tr><th scope="row">Protocol</th><td>Random negatives at 1:5 for all flat evaluations</td></tr>
            <tr><th scope="row">Seeds</th><td>[42, 0, 1] — mean ± std</td></tr>
            <tr>
              <th scope="row">Tables</th>
              <td><code>scripts/make_tables.py</code> reads the result JSONs; nothing typed by hand.</td>
            </tr>
          </tbody>
        </table>
      </section>

      <section className="section" aria-labelledby="leakage">
        <h2 id="leakage">Leakage Check</h2>
        <p>
          Zero-shot results are accepted only after <code>scripts/check_leakage.py</code> passes:
          no held-out test disease may carry a training treatment edge.
        </p>
        <p>
          Output: <code>results/metrics/leakage_check_seed{'{n}'}.json</code>. Status:{' '}
          <strong>PASS</strong> for all seeds.
        </p>
      </section>
    </div>
  )
}
