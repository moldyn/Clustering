#!/usr/bin/env python3

def formatted(html, latex, filetype):
  if filetype == "HTML":
    return html
  elif filetype == "LATEX":
    return latex
  else:
    raise Exception("UNKNOWN FORMAT")

def header(title, filetype):
  html = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>%s</title>
<style type="text/css">
body{
  font-family: Verdana, sans-serif;
  background-color: #FFFFFF;
}
h1, h2, h3, h4 {
  padding: 10px;
  border: 0px;
  margin: 0px;
  text-align: center;
  font-variant: small-caps;
}
h1 {
  background-color: #CCCCCC;
  padding: 25px;
}
h2 {
  background-color: #DDFFDD;
}
div {
  background-color: #EEEEEE;
  padding: 20px;
}
</style>
</head>
<body>
<h1>%s</h1>"""
  latex = """
TODO: %s
  """
  return formatted(html % (title, title)
                 , latex % title
                 , filetype)

def footer(filetype):
  html = "</body></html>"
  latex = "TODO"
  return formatted(html, latex, filetype)

def section(title, filetype):
  html = "<h2>%s</h2>\n"
  latex = "\section{%s}\n"
  return formatted(html % title
                 , latex % title
                 , filetype)

def subsection(title, filetype):
  html = "<h3>%s</h3>\n"
  latex = "\subsection{%s}\n"
  return formatted(html % title
                 , latex % title
                 , filetype)

def emph(content, filetype):
  html = "<i>%s</i>"
  latex = "\emph{%s}"
  return formatted(html % content
                 , latex % content
                 , filetype)

def content(content, filetype):
  html = "<div>%s</div>\n"
  latex = "%s\n"
  return formatted(html % content
                 , latex % content
                 , filetype)

########

def document(filetype):
  clustering = emph("clustering", filetype)
  doc  = header("The " + clustering + " Handbook", filetype)
  doc += section("Installation of the " + clustering + " tool", filetype)
  doc += content("Blah, blah blah.", filetype)
  doc += footer(filetype)
  return doc

print(document("HTML"))

