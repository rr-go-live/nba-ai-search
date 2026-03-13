"""
agent_runner.py
===============
Runs as a GitHub Action step.
Reads a GitHub Issue, sends codebase + issue to Claude,
applies the returned file changes, commits them, opens a PR.
"""

import os
import re
import json
import subprocess
import requests
import anthropic

# ── Config from environment ───────────────────────────────────────────────────
ISSUE_NUMBER  = os.environ["ISSUE_NUMBER"]
REPO          = os.environ["REPO"]           # e.g. "alice/my-app"
GH_TOKEN      = os.environ["GH_TOKEN"]
ANTHROPIC_KEY = os.environ["ANTHROPIC_API_KEY"]

GH_HEADERS = {
    "Authorization": f"Bearer {GH_TOKEN}",
    "Accept": "application/vnd.github+json"
}

# ── Step 1: Fetch the full issue from GitHub API ──────────────────────────────
def fetch_issue():
    url = f"https://api.github.com/repos/{REPO}/issues/{ISSUE_NUMBER}"
    r = requests.get(url, headers=GH_HEADERS)
    r.raise_for_status()
    issue = r.json()

    # Also grab comments so the agent sees the full conversation
    comments_url = issue["comments_url"]
    cr = requests.get(comments_url, headers=GH_HEADERS)
    comments = [c["body"] for c in cr.json()]

    return issue, comments


# ── Step 2: Read the codebase ─────────────────────────────────────────────────
INCLUDE_EXTENSIONS = {".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".css",
                      ".json", ".yaml", ".yml", ".md", ".env.example"}

EXCLUDE_DIRS = {"node_modules", ".git", "__pycache__", ".venv",
                "venv", "dist", "build", ".next"}

MAX_FILES     = 40    # cap to stay within context window
MAX_FILE_BYTES = 30_000  # skip very large files

def read_codebase():
    files = {}
    for root, dirs, filenames in os.walk("."):
        # Prune excluded directories in place
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

        for fname in filenames:
            ext = os.path.splitext(fname)[1]
            if ext not in INCLUDE_EXTENSIONS:
                continue
            path = os.path.join(root, fname)
            size = os.path.getsize(path)
            if size > MAX_FILE_BYTES:
                continue
            try:
                files[path] = open(path, encoding="utf-8", errors="replace").read()
            except Exception:
                pass
            if len(files) >= MAX_FILES:
                return files
    return files


# ── Step 3: Build the prompt and call Claude ──────────────────────────────────
def run_claude(issue, comments, codebase):
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

    # Format codebase as readable blocks
    codebase_text = ""
    for path, content in codebase.items():
        codebase_text += f"\n\n### {path}\n```\n{content}\n```"

    # Format comments if any
    comments_text = ""
    if comments:
        comments_text = "\n\nIssue comments (read these for extra context):\n"
        for i, c in enumerate(comments, 1):
            comments_text += f"\nComment {i}:\n{c}"

    prompt = f"""You are an expert software engineer. Your job is to complete a GitHub Issue by writing code.

Issue #{ISSUE_NUMBER}: {issue['title']}

Issue description:
{issue['body']}
{comments_text}

Current codebase:
{codebase_text}

INSTRUCTIONS:
1. Read the issue carefully. Understand exactly what needs to change.
2. Look at the existing code to understand patterns, style, and structure.
3. Implement the minimum changes needed to satisfy the issue.
4. Do not refactor things that aren't related to the issue.
5. Match the existing code style exactly.

OUTPUT FORMAT — respond with ONLY a valid JSON object, no other text:
{{
  "analysis": "2-3 sentences explaining what you understood from the issue and what you will change",
  "files_changed": {{
    "path/to/file.py": "complete new file content here — the ENTIRE file, not just the changed parts",
    "path/to/new_file.py": "content of any new file"
  }},
  "files_deleted": [
    "path/to/file/to/delete.py"
  ],
  "commit_message": "type(scope): short description — Closes #{ISSUE_NUMBER}",
  "pr_title": "Short PR title",
  "pr_body": "## What changed\\n\\nDescription of changes.\\n\\n## How to test\\n\\nSteps to verify.\\n\\nCloses #{ISSUE_NUMBER}"
}}

If the issue is unclear or cannot be completed safely, return:
{{
  "error": "explanation of why you cannot complete this"
}}"""

    response = client.messages.create(
        model="claude-opus-4-20250514",
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    start = raw.find('{')
    end   = raw.rfind('}') + 1
    raw   = raw[start:end]

    raw = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', raw)

    try:
        return json.loads(raw, strict=False)
    except json.JSONDecodeError:
        raw = re.sub(r'(?<!\\)\n', '\\n', raw)
        return json.loads(raw, strict=False)

# ── Step 4: Apply file changes ────────────────────────────────────────────────
def apply_changes(result):
    for path, content in result.get("files_changed", {}).items():
        # Create parent directories if needed
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  Written: {path}")

    for path in result.get("files_deleted", []):
        if os.path.exists(path):
            os.remove(path)
            print(f"  Deleted: {path}")


# ── Step 5: Commit, push, open PR ─────────────────────────────────────────────
def git(cmd):
    """Run a git command, raise on failure."""
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"git command failed: {cmd}\n{result.stderr}")
    return result.stdout.strip()


def commit_and_pr(result):
    branch = f"agent/issue-{ISSUE_NUMBER}"

    git(f"git checkout -b {branch}")
    git("git add -A")
    git(f'git commit -m "{result["commit_message"]}"')
    git(f"git push origin {branch}")

    # Open the PR via GitHub API
    pr_payload = {
        "title": result["pr_title"],
        "body":  result["pr_body"],
        "head":  branch,
        "base":  "main",          # change to "master" if your default is master
        "draft": True             # opens as draft — you review before merging
    }
    r = requests.post(
        f"https://api.github.com/repos/{REPO}/pulls",
        headers=GH_HEADERS,
        json=pr_payload
    )
    r.raise_for_status()
    pr_url = r.json()["html_url"]
    print(f"\nPR opened: {pr_url}")
    return pr_url


# ── Step 6: Comment on the issue ──────────────────────────────────────────────
def comment_on_issue(pr_url, analysis):
    body = (
        f"**Agent started work on this issue.**\n\n"
        f"**Analysis:** {analysis}\n\n"
        f"**Draft PR:** {pr_url}\n\n"
        f"Please review the PR before merging."
    )
    requests.post(
        f"https://api.github.com/repos/{REPO}/issues/{ISSUE_NUMBER}/comments",
        headers=GH_HEADERS,
        json={"body": body}
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Fetching issue #{ISSUE_NUMBER} from {REPO}...")
    issue, comments = fetch_issue()

    print("Reading codebase...")
    codebase = read_codebase()
    print(f"  Loaded {len(codebase)} files")

    print("Running Claude agent...")
    result = run_claude(issue, comments, codebase)

    if "error" in result:
        print(f"Agent declined: {result['error']}")
        requests.post(
            f"https://api.github.com/repos/{REPO}/issues/{ISSUE_NUMBER}/comments",
            headers=GH_HEADERS,
            json={"body": f"**Agent could not complete this issue:**\n\n{result['error']}"}
        )
        return

    print(f"Analysis: {result['analysis']}")
    print("Applying changes...")
    apply_changes(result)

    print("Committing and opening PR...")
    pr_url = commit_and_pr(result)

    comment_on_issue(pr_url, result["analysis"])
    print("Done.")


if __name__ == "__main__":
    main()