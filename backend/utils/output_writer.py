import os
import tempfile
from backend.modules import report_generator

def sanitize_filename(name):
    return "_".join(name.strip().lower().split())

def save_summary_to_file(summaries, goal, format="md"):
    safe_goal = sanitize_filename(goal)
    filename = f"{safe_goal}_summary_report.{format}"
    output_path = os.path.join(tempfile.gettempdir(), filename)

    content = (
        report_generator.generate_markdown_report(summaries, goal)
        if format == "md"
        else report_generator.generate_report(summaries, goal, return_as_string=True)
    )

    if isinstance(content, list):
        content = "\n".join(content)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"✅ Summary written to: {output_path}")
    
    # ✅ Return as download endpoint URL
    return f"/download?path={output_path}"