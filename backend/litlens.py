import os
import argparse

from backend.modules import pdf_extractor, embedder, relevance_filter, summarizer, report_generator

# ----------------------
# Argument Parser
# ----------------------
parser = argparse.ArgumentParser(description="LitLens - Literature Review Assistant")

parser.add_argument("--goal", type=str, required=True, help="Your research goal")
parser.add_argument("--input-dir", type=str, default="test_pdfs", help="Directory containing PDF papers")
parser.add_argument("--threshold", type=float, default=0.4, help="Relevance threshold (default=0.4)")
parser.add_argument("--output", type=str, default="litreview", help="Output directory for the generated report")
parser.add_argument("--format", type=str, choices=["txt", "md"], default="txt", help="Output format: txt (default) or md")
parser.add_argument("--verbose", action="store_true", help="Enable detailed logs (Step 7 ready)")

args = parser.parse_args()

# ----------------------
# Extract CLI Arguments
# ----------------------
goal = args.goal
input_dir = args.input_dir
threshold = args.threshold
output_dir = args.output
output_format = args.format
verbose = args.verbose

# ----------------------
# Create Output Directory
# ----------------------
os.makedirs(output_dir, exist_ok=True)

# ----------------------
# Step 1 - Extract PDFs
# ----------------------
if verbose:
    print("📄 Extracting papers...")

papers = pdf_extractor.extract_papers(input_dir)

# ----------------------
# Step 2 - Embed Goal & Papers
# ----------------------
if verbose:
    print("📌 Embedding goal and papers...")

goal_embedding, paper_embeddings = embedder.embed_goal_and_papers(goal, papers)

# ----------------------
# Step 3 - Filter Relevant Papers
# ----------------------
if verbose:
    print("🎯 Filtering relevant papers...")

relevant_indices = relevance_filter.filter_relevant_papers(goal_embedding, paper_embeddings, threshold=threshold)
relevant_papers = [papers[i] for i in relevant_indices]

print(f"[LitLens] Selected {len(relevant_papers)} relevant papers")

# ----------------------
# Step 4 - Summarize Papers
# ----------------------
if verbose:
    print("🧠 Summarizing papers...")

summaries = summarizer.summarize_papers(relevant_papers, goal)

# ----------------------
# Step 5 - Generate Report
# ----------------------
if verbose:
    print("📝 Generating report...")

output_path = os.path.join(output_dir, f"litlens_summary_report.{output_format}")
report_generator.generate_report(summaries, goal, output_path=output_path, format=output_format)

print("✅ LitLens Report Generation Complete.")
