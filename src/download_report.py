from docx import Document
from fpdf import FPDF


class DownloadReport:

    def save_text_as_file(self, report, file_format, file_name="report"):
        """"Saving the text as per the user's choice"""
        if file_format.lower() == 'pdf':
            self.save_as_pdf(report, file_name)
        elif file_format.lower() == 'docx':
            self.save_as_docx(report, file_name)
        elif file_format.lower() == 'html':
            self.save_as_html(report, file_name)
        elif file_format.lower() == 'txt':
            self.save_as_txt(report, file_name)
        else:
            print("Unsupported file format. Please choose from 'pdf', 'docx', 'html', or 'txt'.")

    def save_as_pdf(self, text, file_name):
        """"Saving the text in the PDF file format"""

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.write(5, text)
        pdf.output(file_name + ".pdf", "F")

    def save_as_docx(self, text, file_name):
        """"Saving the text in the Docx file format"""

        doc = Document()
        doc.add_paragraph(text)
        doc.save(file_name + '.docx')

    def save_as_html(self, text, file_name):
        """"Saving the text in the HTML file format"""

        html_content = ""
        paragraphs = text.split("\n")
        for paragraph in paragraphs:
            html_content += f"<p>{paragraph}</p>"
        with open(file_name + ".html", 'w', encoding="utf-8") as f:
            f.write(f"<!DOCTYPE html>\n<html>\n<body>\n{html_content}\n</body>\n</html>")

    def save_as_txt(self, text, file_name):
        """"Saving the text in the TXT file format"""

        with open(file_name + '.txt', 'w') as file:
            file.write(text)
