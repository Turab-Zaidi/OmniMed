import re

def clean_report(text):
    text = text.lower()
    
    findings = re.search(r'findings:(.*?)(impression:|indication:|comparison:|$)', text, re.DOTALL)
    impression = re.search(r'impression:(.*?)(indication:|comparison:|$)', text, re.DOTALL)
    
    extracted_text = ""
    if findings:
        extracted_text += "Findings: " + findings.group(1).strip() + " "
    if impression:
        extracted_text += "Impression: " + impression.group(1).strip()
        
    if not extracted_text:
        return text.strip()
        
    return extracted_text.strip()