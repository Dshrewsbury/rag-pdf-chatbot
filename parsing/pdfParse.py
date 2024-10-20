import copy
import re
import logging
from pypdf import PdfReader as pdf2_read
import numpy as np


def tokenize(d, t):
    d["content_with_weight"] = t
    t = re.sub(r"</?(table|td|caption|tr|th)( [^<>]{0,12})?>", " ", t)
    d["content_ltks"] = rag_tokenizer.tokenize(t)
    d["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(d["content_ltks"])


def tokenize_table(tbls, doc, batch_size=10):
    res = []
    # add tables
    for (img, rows), poss in tbls:
        if not rows:
            continue
        if isinstance(rows, str):
            d = copy.deepcopy(doc)
            tokenize(d, rows)
            d["content_with_weight"] = rows
            if img: d["image"] = img
            #if poss: add_positions(d, poss)
            res.append(d)
            continue
        de = "; "
        for i in range(0, len(rows), batch_size):
            d = copy.deepcopy(doc)
            r = de.join(rows[i:i + batch_size])
            tokenize(d, r)
            d["image"] = img
            #add_positions(d, poss)
            res.append(d)
    return res

def tokenize_chunks(chunks, doc, eng, pdf_parser=None):
    res = []
    # wrap up as es documents
    for ck in chunks:
        if len(ck.strip()) == 0:continue
        print("--", ck)
        d = copy.deepcopy(doc)
        if pdf_parser:
            try:
                d["image"], poss = pdf_parser.crop(ck, need_position=True)
                #add_positions(d, poss)
                ck = pdf_parser.remove_tag(ck)
            except NotImplementedError as e:
                pass
        tokenize(d, ck, eng)
        res.append(d)
    return res

class PlainParser(object):
    def __call__(self, filename, from_page=0, to_page=100000, **kwargs):
        self.outlines = []
        lines = []
        try:
            self.pdf = pdf2_read(filename)
            for page in self.pdf.pages[from_page:to_page]:
                lines.extend([t for t in page.extract_text().split("\n")])

            outlines = self.pdf.outline

            def dfs(arr, depth):
                for a in arr:
                    if isinstance(a, dict):
                        self.outlines.append((a["/Title"], depth))
                        continue
                    dfs(a, depth + 1)

            dfs(outlines, 0)
        except Exception as e:
            logging.warning(f"Outlines exception: {e}")
        if not self.outlines:
            logging.warning(f"Miss outlines")

        return [(l, "") for l in lines], []

    def crop(self, ck, need_position):
        raise NotImplementedError

    @staticmethod
    def remove_tag(txt):
        raise NotImplementedError


def chunk(filename, from_page=0, to_page=100000, callback=None, **kwargs):
    """
    Only PDF is supported.
    The abstract of the paper will be sliced as an entire chunk, and will not be sliced partly.
    """
    if re.search(r"\.pdf$", filename, re.IGNORECASE):
        pdf_parser = PlainParser()
        paper = {
            "title": filename,
            "authors": " ",
            "abstract": "",
            "sections": pdf_parser(filename, from_page=from_page, to_page=to_page)[0],
            "tables": []
        }
    else:
        raise NotImplementedError("file type not supported yet(pdf supported)")

    doc = {
        "docnm_kwd": filename,
        "authors_tks": rag_tokenizer.tokenize(paper["authors"]),
        "title_tks": rag_tokenizer.tokenize(paper["title"] if paper["title"] else filename)
    }
    doc["title_sm_tks"] = rag_tokenizer.fine_grained_tokenize(doc["title_tks"])
    doc["authors_sm_tks"] = rag_tokenizer.fine_grained_tokenize(doc["authors_tks"])

    # Tokenize tables
    #res = tokenize_table(paper["tables"], doc, eng)
    res = []

    # Tokenize abstract if available
    if paper["abstract"]:
        d = copy.deepcopy(doc)
        txt = paper["abstract"]
        d["important_kwd"] = ["abstract", "summary"]
        d["important_tks"] = " ".join(d["important_kwd"])
        tokenize(d, txt)
        res.append(d)

    # Process and tokenize sections
    sorted_sections = paper["sections"]
    bull = bullets_category([txt for txt, _ in sorted_sections])
    most_level, levels = title_frequency(bull, sorted_sections)

    sec_ids = []
    sid = 0
    for i, lvl in enumerate(levels):
        if lvl <= most_level and i > 0 and lvl != levels[i - 1]:
            sid += 1
        sec_ids.append(sid)

    chunks = []
    last_sid = -2
    for (txt, _), sec_id in zip(sorted_sections, sec_ids):
        if sec_id == last_sid:
            if chunks:
                chunks[-1] += "\n" + txt
                continue
        chunks.append(txt)
        last_sid = sec_id

    # Tokenize the chunks
    res.extend(tokenize_chunks(chunks, doc, pdf_parser))
    return res


if __name__ == "__main__":
    import sys

    # Dummy callback function
    def dummy(prog=None, msg=""):
        pass

    # Process the PDF file passed via command line
    chunk(sys.argv[1], callback=dummy)

