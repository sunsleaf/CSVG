import re
import textwrap


def validate_program(prog: str):
    """
    validate the input program (@prog).
    if the program is valid or can be corrected, return a clean and corrected version of it.
    if the program contains fatal errors, return all errors.
    """
    # extract python code if the llm output is in markdown format (which happens from time to time...)
    if "```" in prog:
        if prog.count("```") != 2:
            print()
            print("=>> ↓↓↓ PROGRAM VALIDATION FAILED! ↓↓↓ <<=")
            print()
            print(prog)
            print()
            print("=>> ↑↑↑ PROGRAM VALIDATION FAILED! ↑↑↑ <<=")
            print()
            return textwrap.dedent(prog).strip()

        is_code = False
        code_lines = []

        for line in prog.split("\n"):
            match = re.match(
                r"(?:```\s*python|```\s*Python|```\s*PYTHON|```)(.*?)$",
                line.strip(),
            )
            if match:
                is_code = not is_code
                continue  # simply skip this line

            if is_code:
                # assuming no indent in the code
                code_lines.append(line.strip())

        prog = "\n".join(code_lines)

    prog = textwrap.dedent(prog).strip()

    # print("cleaned program:")
    # print()
    # print(prog)
    # print()

    # maybe we don't need a validator for now...

    return prog
