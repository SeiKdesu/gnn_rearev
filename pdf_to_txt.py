from pypdf import PdfReader

input_pdf = "Gao et al. - 2025 - D-RAG Differentiable Retrieval-Augmented Generation for Knowledge Graph Question Answering.pdf"   # 変換したいPDFファイル名
output_txt = "DRAG.txt"  # 出力したいテキストファイル名

reader = PdfReader(input_pdf)

all_text = []
for page in reader.pages:
    text = page.extract_text()
    if text:
        all_text.append(text)

joined_text = "\n\n".join(all_text)

with open(output_txt, "w", encoding="utf-8") as f:
    f.write(joined_text)

print(f"完了: {output_txt} に保存しました")
