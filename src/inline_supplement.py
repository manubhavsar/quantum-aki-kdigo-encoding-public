"""Inline paper/supplement_body.tex into paper/submission.tex after the response."""
p = "paper/submission.tex"
s = open(p).read()
body = open("paper/supplement_body.tex").read()
n = body.count("\\begin{table}")
BEGIN, END = "% ---- begin supplementary material (generated) ----", "% ---- end supplementary material ----"
block = (BEGIN + "\n\\ifsupplement\n\\clearpage\n\\setcounter{page}{1}\n\\renewcommand{\\thepage}{S\\arabic{page}}\n"
         "\\setcounter{table}{0}\n\\renewcommand{\\thetable}{S\\arabic{table}}\n"
         "\\section*{Supplementary material}\n\\noindent Tables S1--S" + str(n) + " support the manuscript and the "
         "response; every value is read from the experiment logs in the anonymous repository.\n\n"
         + body + "\n\\fi\n" + END)
if BEGIN in s:
    a, b = s.index(BEGIN), s.index(END) + len(END)
    s = s[:a] + block + s[b:]
else:
    s = s.replace("\\end{document}", block + "\n\n\\end{document}")
open(p, "w").write(s)
print("supplement inlined")
