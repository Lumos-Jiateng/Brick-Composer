"""
prompt.py
---------
All prompt content lives here.  Edit SYSTEM_PROMPT and build_user_message()
to change what the model sees without touching any other file.
"""

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a LEGO brick assembly expert.

You will be given three images:
  1. **Original views**: 6 orthogonal views (Back, Bottom, Left, Right, Up, Front) of the object before the assembly step of a certain brick.
  2. **Target views**: the same 6 orthogonal views of the target object after the brick was installed, with the brick highlighted with a bounding box.
  3. **Remaining brick catalog**: a grid showing every candidate brick that still needs to be placed in and can be selected for the current assembly step, with part filenames (e.g. 3023.dat) labeled above each image.

Your task:
  Compare the Original and Target views carefully to identify which brick(s) must be placed, \
then locate each one in the catalog.

Output format — one line per brick, exactly as shown below:
  <part_filename>, row <R>, col <C>

Rules:
  • <part_filename> must be the exact .dat name shown in the catalog label (e.g. 3023.dat).
  • <R> and <C> are the 1-based row and column of that brick in the catalog grid.
  • If the same brick appears more than once in a step, repeat the line (with the \
position of each individual tile).
  • Do NOT output any explanation, preamble, or extra text — only the formatted lines.\
"""

# ---------------------------------------------------------------------------
# User message builder
# ---------------------------------------------------------------------------

def build_user_message(
    original_b64: str,
    target_b64: str,
    catalog_b64: str,
    is_module: bool = False,
) -> list[dict]:
    """
    Build the user-turn content list for the OpenAI chat API.

    Args:
        original_b64 : base64-encoded PNG of the 3×2 grid of original views.
        target_b64   : base64-encoded PNG of the 3×2 grid of target  views.
        catalog_b64  : base64-encoded PNG of the remaining-brick catalog.
        is_module    : True for module steps (multiple bricks); False for
                       single-brick steps.  Only the closing question changes.

    Returns:
        List of content blocks (text + image_url) ready for the messages array.
    """
    if is_module:
        closing = (
            "This step places a module made of multiple bricks.\n"
            "Output one line per brick in the format:  <part.dat>, row <R>, col <C>\n"
            "Repeat a line if the same part appears more than once."
        )
    else:
        closing = (
            "This step places exactly ONE brick.\n"
            "Output a single line in the format:  <part.dat>, row <R>, col <C>"
        )

    return [
        {
            "type": "text",
            "text": (
                "Below are the three inputs for this assembly step.\n\n"
                "### Image 1 — Original views (brick not yet placed, 6 orthogonal angles):"
            ),
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{original_b64}"},
        },
        {
            "type": "text",
            "text": "### Image 2 — Target views (brick in place, 6 orthogonal angles):",
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{target_b64}"},
        },
        {
            "type": "text",
            "text": (
                "### Image 3 — Remaining brick catalog "
                "(rows and columns are 1-indexed from the top-left; "
                "part names are labeled above each tile):"
            ),
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{catalog_b64}"},
        },
        {
            "type": "text",
            "text": closing,
        },
    ]
