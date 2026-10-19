# URA Right-Law Retrieval: Experiment Notebook

Notebook dùng để chạy và so sánh các giả thuyết (H1–H11) về **truy hồi điều khoản pháp luật tiếng Việt**: BM25, dense (bge-m3), hybrid, reranking, các cách cắt văn bản (Điều/Khoản/Điểm, đoạn cố định, cha–con) và lọc theo thời điểm hiệu lực.

Toàn bộ code nằm trong [`legal_retrieval_experiments.ipynb`](legal_retrieval_experiments.ipynb). Đây là bản gộp các module của repo Workstream B (`URA_Right-Law-Retrieval_Workstream-B`, mỗi ô ghi tên file gốc tương ứng) và có thể chạy độc lập. Mỗi giả thuyết là một file YAML trong [`configs/`](configs/).

## Cấu trúc thư mục

```
.
├── legal_retrieval_experiments.ipynb   # toàn bộ code + các bước chạy
├── configs/
│   ├── base.yaml                       # cấu hình gốc (= baseline H1)
│   ├── hypothesis_template.yaml        # mẫu để tạo giả thuyết mới
│   ├── examples/h*.yaml                # H1–H11
│   └── suites/*.yaml                   # nhóm giả thuyết chạy cùng lúc
├── data/
│   ├── enriched/*_enriched.json        # kho văn bản từ Workstream A (chỉ đọc)
│   └── benchmarks/*.jsonl              # bộ câu hỏi đánh giá (Workstream C)
└── artifacts/                          # kết quả chạy (tự sinh, không commit)
```

## Bắt đầu nhanh

Yêu cầu Python ≥ 3.10 (đã thử với 3.12). Mọi thứ cài trong venv của thư mục này.

```bash
python3 -m venv .venv
.venv/bin/pip install numpy PyYAML pandas ipykernel              # đủ cho BM25: H1, H6–H11
.venv/bin/pip install "sentence-transformers>=3.0" "faiss-cpu>=1.8.0"   # thêm nếu chạy dense: H2–H5
```

1. Mở notebook trong VS Code/Jupyter và chọn kernel `.venv`.
2. Ở ô cấu hình (mục 0), chỉnh:
   - `BENCHMARK_PATH`: file câu hỏi dùng chung cho mọi giả thuyết.
   - `RUN_DENSE`: `False` chỉ chạy BM25, mất vài giây; `True` chạy thêm H2–H5, cần tải model và nên có GPU.
3. **Run All**, hoặc chỉ chạy những mục cần (xem [Các mục trong notebook](#các-mục-trong-notebook)).

Model Hugging Face (`BAAI/bge-m3`, `BAAI/bge-reranker-v2-m3`, tổng khoảng 6,4 GB) được tải tự động vào `.hf_cache/` trong thư mục này, không vào `~/.cache`.

## Dữ liệu

**Kho văn bản** (`data/enriched/`): 11 văn bản về đất đai, nhà ở và kinh doanh bất động sản. Mỗi phần tử là một đơn vị (văn bản, chương, Điều, Khoản hoặc Điểm) với các trường `unit_id`, `text`, `citation_surface`, `title_chain`, `effective_from`/`effective_to`… Mặc định chỉ dùng Điều/Khoản/Điểm, còn 9.853 đơn vị sau khi xử lý trùng. Code không bao giờ sửa dữ liệu này.

**Đơn vị bị trùng `unit_id`:** dữ liệu gốc có 1.554 dòng trùng. Với `corpus.duplicate_unit_id_policy: dedupe`:
- Dòng giống hệt nhau bị bỏ.
- Dòng trùng ID nhưng khác nội dung vẫn được giữ, với `record_id` có hậu tố `#dupN`. Khi chấm điểm, chúng vẫn được tính là cùng một đơn vị luật.

**Benchmark** (`data/benchmarks/`), mỗi dòng một câu hỏi:

```json
{"query_id": "example-001",
 "query": "Nguyên tắc đo đạc lập bản đồ địa chính là gì?",
 "relevant_unit_ids": ["101-2024-NĐ-CP__article-3__clause-1__point-a__v01/08/2024"],
 "filters": {"doc_id": "101-2024-NĐ-CP"}}
```

`relevant_unit_ids` phải là `unit_id` có thật trong kho; mục 3 của notebook có kiểm tra điều này. `filters` là tùy chọn. Lưu ý: lọc theo `doc_id` đáp án làm bài toán dễ hơn thực tế.

## Pipeline

```
config YAML → load_config (gộp chuỗi extends + kiểm tra biến)
           → build_records (corpus → record)
           → chỉ mục BM25 / FAISS
với mỗi câu hỏi:
  xử lý câu hỏi → lọc metadata + thời gian → tìm sparse/dense
  → gộp kết quả (RRF / có trọng số) → xếp hạng lại → bỏ trùng → top-k
  → chấm điểm + đo thời gian/RAM/GPU → ghi artifacts/runs/<H>/<run_id>/
```

**Corpus → record.** BM25 và bge-m3 chỉ nhận một chuỗi văn bản cho mỗi phần tử trong chỉ mục. Bước `build_records` quyết định ba việc:
1. Đơn vị tìm kiếm là gì: mỗi Điều/Khoản/Điểm, đoạn 220 từ, hay Khoản/Điểm con.
2. Ghép những trường nào vào chuỗi: `[CITATION]`, `[TITLE]`, `[TEXT]`, `[PARENT]`.
3. Mỗi record thuộc đơn vị luật nào (`source_unit_ids`), để chấm điểm.

Đây là biến thí nghiệm của H1 và H5–H9, nên được điều khiển qua `representation.*` trong config chứ không cố định từ trước.

## Các giả thuyết

Mỗi giả thuyết `extends` một config cha (baseline trực tiếp của nó) và chỉ được đổi các biến đã khai báo trong `hypothesis.variables`. `load_config` **báo lỗi** nếu có thay đổi nằm ngoài danh sách đó.

```
H1  baseline: BM25 trên từng Điều/Khoản/Điểm
├── H2   dense (bge-m3) thay BM25
├── H3   hybrid BM25 + dense, gộp RRF
│   ├── H4   + cross-encoder reranker (bge-reranker-v2-m3)
│   └── H5   đoạn cố định 220 từ (hybrid)
├── H6   đoạn cố định 220 từ (BM25)
├── H7   chỉ Khoản/Điểm, không có bản ghi Điều
│   └── H8   + tiêu đề Điều ghép vào Khoản/Điểm
│       └── H9   tìm Điều trước, rồi xếp hạng Khoản/Điểm bên trong
│           └── H11  + lọc thời gian
└── H10  + lọc thời gian
```

**Baseline H1** (= `configs/base.yaml`):
- Dữ liệu: Điều/Khoản/Điểm, bỏ trùng.
- Văn bản đưa vào chỉ mục: `[CITATION] + [TITLE] + [TEXT]`.
- Câu hỏi: chuẩn hóa, chữ thường, bỏ dấu tiếng Việt.
- Tìm kiếm: BM25 (k1 = 1.5, b = 0.75), lấy top 50 ứng viên, trả top 10, bỏ trùng theo đơn vị luật.
- Không dense, không reranker, không lọc thời gian.
- Seed 42; model ghim theo commit (`reproducibility_mode: final`).

**Lọc thời gian** (H10, H11):
- Câu hỏi có mốc thời gian ("tại thời điểm 01/06/2025", "trước năm 2020"…): chỉ giữ điều khoản còn hiệu lực trong khoảng đó.
- Câu hỏi không có mốc: hiểu là "hiện hành" tại `temporal.reference_date` (đang ghim 2026-09-19).

## Đánh giá

Các chỉ số: hit_rate, precision, recall, MRR, nDCG, MAP tại k = 1, 3, 5, 10, 20, mỗi chỉ số ở **hai cấp**:

| Cấp | Tên cột | Tính trúng khi |
|---|---|---|
| Chặt | `recall@10`… | Kết quả chứa đúng đơn vị đáp án (`match_by: source_unit`) |
| Điều | `article.recall@10`… | Kết quả nằm trong cùng phiên bản Điều với đáp án (`also_match_by: [article]`) |

Cấp Điều giúp so sánh công bằng giữa các cách cắt văn bản. Ví dụ: H1 trả về cả Điều 3 thì không trúng ở cấp chặt nhưng trúng ở cấp Điều; H7 không có bản ghi Điều nên không bao giờ trúng một đáp án cấp Điều.

Mỗi đơn vị đáp án chỉ được tính ở lần trúng đầu tiên, nên các đoạn chồng lấn không làm tăng điểm giả.

## Kết quả

```
artifacts/
├── reports/
│   ├── summary.csv          # mỗi giả thuyết một dòng (lần chạy mới nhất)
│   └── vs_baseline.csv      # mỗi giả thuyết trừ baseline của nó
└── runs/<H>/<run_id>/       # mỗi lần chạy một thư mục, run_id = thời gian UTC + mã ngắn
    ├── config.yaml          # config đầy đủ đã chạy (sau khi gộp extends)
    ├── summary.json         # kết quả + thông tin lần chạy
    └── per_query.jsonl      # kết quả từng câu hỏi
```

**Muốn so sánh nhanh:** mở `reports/summary.csv` và `reports/vs_baseline.csv`. Hai file chỉ lấy chỉ số chính (`recall`, `mrr`, `ndcg` @10, cả cấp chặt lẫn cấp Điều) cùng thời gian chạy.

| File | Cột / trường chính |
|---|---|
| `summary.csv` | `hypothesis`, `baseline`, 6 chỉ số chính, `queries`, `records`, `query_mean_ms`, `index_seconds`, `peak_gpu_gb`, `run_id` |
| `vs_baseline.csv` | `hypothesis`, `baseline`, `changed` (biến đã đổi), `Δ <chỉ số>` = giả thuyết − baseline (dương là tốt hơn; riêng `Δ query_mean_ms` dương là chậm hơn) |
| `summary.json` | `hypothesis`, `baseline`, `changed`, `metrics` (bảng đủ: `strict`/`article` → chỉ số → `@k`), `timing` (dựng chỉ mục, ms mỗi câu), `resources` (đỉnh RAM/VRAM theo GB), `environment` (seed, phiên bản thư viện, commit model, git) |
| `per_query.jsonl` | Mỗi dòng một câu hỏi: `first_hit_rank` (hạng của kết quả đúng đầu tiên, `strict`/`article`), `metrics` (6 chỉ số chính), `latency_ms`, `results` (mỗi kết quả: `rank`, `record_id`, `score`, `hit`, `article_hit`, `snippet` 160 ký tự) |

Chỉ số chính được đặt ở đầu mục 1.17 của notebook (`HEADLINE_K = 10`, `HEADLINE_METRICS`). Lần chạy cũ không bị ghi đè: `summary.csv` lấy lần mới nhất của mỗi giả thuyết, còn `compare_experiments(latest_only=False)` liệt kê tất cả.

Trong notebook (mục 7):
- `compare_experiments()` ghi và hiển thị hai bảng trên.
- `compare_queries("H3", "H4")` in hạng kết quả đúng đầu tiên của từng câu hỏi ở hai giả thuyết, để xem câu nào tốt lên hoặc kém đi.
- `show_metrics(run_dir)` in bảng chỉ số đầy đủ của một lần chạy.

## Các mục trong notebook

| Mục | Nội dung | Thời gian |
|---|---|---|
| 0 | Cấu hình, import | — |
| 1.1–1.18 | Định nghĩa module: text, temporal, config, corpus, query, encoder, BM25, FAISS, index manager, fusion, rerank, pipeline, evaluation, runtime, runner, helpers | vài giây |
| 2 | Unit check nhanh + nạp và kiểm tra H1–H11 | vài giây |
| 3 | Thống kê kho văn bản, kiểm tra trùng, thống kê record, kiểm tra benchmark | vài giây |
| 4 | Thử một câu hỏi (không ghi file) | vài giây |
| 5 | Chạy một giả thuyết + xem từng câu hỏi | tùy giả thuyết |
| 6 | Chạy cả bộ H1–H11 | khoảng 30 phút nếu `RUN_DENSE=True` |
| 7 | `summary.csv`, `vs_baseline.csv`, so sánh từng câu hỏi giữa hai giả thuyết | vài giây |

Chỉ muốn xem kết quả đã có: chạy mục 0–1 rồi nhảy xuống mục 7. Muốn chạy một giả thuyết: đổi `"H1"` ở mục 5 thành ID cần chạy.

## Thêm giả thuyết mới

1. Chép `configs/hypothesis_template.yaml` vào `configs/examples/`.
2. Đặt `extends` trỏ tới baseline (ví dụ `h3_hybrid_rrf.yaml`), đặt `hypothesis.id` và mô tả.
3. Chỉ ghi đè các biến cần đổi và khai báo đúng các biến đó trong `hypothesis.variables`.
4. Thêm vào `CONFIG_FILES` (mục 2 của notebook) và vào file suite trong `configs/suites/`.

```yaml
extends: h3_hybrid_rrf.yaml
hypothesis:
  id: H12
  description: "H3 với trọng số RRF nghiêng về dense."
  variables:
    - fusion.dense_weight
indexing:
  output_dir: artifacts/indexes/h12_dense_weighted_rrf
fusion:
  dense_weight: 0.8
```

## Hạn chế hiện tại

- **Benchmark mới có 1 câu hỏi mẫu.** Các số liệu hiện tại chỉ xác nhận pipeline chạy đúng, chưa đủ để kết luận giả thuyết nào tốt hơn. Cần bộ câu hỏi của Workstream C, nên chia tập dev/test.
- **Chưa có kiểm định thống kê** (bootstrap CI, paired test) giữa các cặp giả thuyết.
- **H5/H6 dễ được tính trúng hơn ở cấp chặt:** mỗi đoạn 220 từ bao phủ nhiều đơn vị, trong khi lượng văn bản trả về không được cân bằng giữa các giả thuyết.
- **H4 trên CPU rất chậm:** benchmark vài trăm câu có thể mất hàng giờ.
- **Dữ liệu Workstream A có 206 `unit_id` bị gán cho nội dung khác nhau** (có vẻ là lỗi parse các mục lồng nhau). Code đang tạm xử lý; nên sửa ở nguồn.
- BM25 tách theo âm tiết, chưa tách từ tiếng Việt.

## So với repo Workstream B

Thư mục `configs/` được chép từ Workstream B rồi chỉnh lại:
- H2 và H3 `extends` trực tiếp H1 (trước là `base.yaml`; giá trị không đổi).
- **H7–H9 tắt lọc thời gian.** H7 cũ đổi cùng lúc cách cắt văn bản và lọc thời gian, nên không tách được tác dụng của từng biến. Thêm **H10** (H1 + lọc thời gian) và **H11** (H9 + lọc thời gian, tức H9 cũ).
- Thêm `corpus.duplicate_unit_id_policy`, `evaluation.also_match_by: [article]`; đổi `indexing.dense.batch_size` thành 2 và reranker của H4 chạy trên CPU.

Notebook cũng có thay đổi tương ứng: xử lý trùng trong `load_enriched_units`, chấm điểm cấp Điều trong `evaluation`, và định dạng kết quả gọn hơn. Mỗi lần chạy ghi 3 file thay vì 5; `per_query.jsonl` không còn lưu toàn văn và metadata của từng kết quả. Phần tổng hợp gồm `summary.csv` + `vs_baseline.csv`, thay cho `experiment_comparison.csv` 80 cột.
