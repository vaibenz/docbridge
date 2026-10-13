"""Generates the sample enterprise corpus for Harbourline Logistics, a fictional company.

Writes PDFs and CSVs into corpus/ plus a manifest that sets which groups may read
each source. Run once: python scripts/make_corpus.py
"""

import csv
import json
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parents[1] / "corpus"

DOCS = {
    "employee_handbook.pdf": {
        "groups": ["all"],
        "title": "Harbourline Logistics Employee Handbook",
        "pages": [
            [
                ("Working hours", "Standard office hours are 9am to 6pm, Monday to Friday, with a one hour lunch break. "
                 "Warehouse staff work rostered shifts published two weeks in advance on the workforce portal. "
                 "Employees may request flexible start times between 7am and 10am with their manager's approval."),
                ("Remote work", "Office based employees may work remotely up to two days per week. Remote days must be "
                 "recorded in the workforce portal by Friday of the previous week. Warehouse, fleet and customs roles are "
                 "not eligible for remote work because they require on site presence."),
            ],
            [
                ("Annual leave", "Full time employees receive 18 days of annual leave per calendar year, rising to 21 days "
                 "after five years of service. Up to 5 unused days may be carried forward to the next year and must be "
                 "used by 31 March, after which they lapse. Leave requests of more than 5 consecutive days need two weeks notice."),
                ("Sick leave", "Employees receive 14 days of outpatient sick leave and 60 days of hospitalisation leave per "
                 "year, inclusive of the outpatient days. A medical certificate is required for any absence longer than one day."),
                ("Parental leave", "Eligible mothers receive 16 weeks of paid maternity leave and eligible fathers receive "
                 "4 weeks of paid paternity leave. Parental leave must be applied for at least 4 weeks before it starts."),
            ],
            [
                ("Code of conduct", "Employees must not accept gifts or hospitality worth more than SGD 100 from customers or "
                 "suppliers. Any gift above that value must be declared to Compliance within 5 working days using the gifts register."),
                ("Reporting concerns", "Concerns about misconduct can be raised with a manager, with HR, or anonymously through "
                 "the Speak Up hotline. Harbourline does not tolerate retaliation against anyone who raises a concern in good faith."),
            ],
        ],
    },
    "it_security_policy.pdf": {
        "groups": ["all"],
        "title": "IT Security Policy",
        "pages": [
            [
                ("Passwords", "Passwords must be at least 14 characters long and must not reuse any of the previous 10 passwords. "
                 "Passwords for standard accounts expire every 180 days. Multi factor authentication is mandatory for email, "
                 "VPN and the warehouse management system."),
                ("Device security", "Company laptops must use full disk encryption and must lock automatically after 5 minutes "
                 "of inactivity. Personal devices may access email only through the managed mobile app."),
            ],
            [
                ("Reporting a security incident", "Suspected phishing, malware or lost devices must be reported to the "
                 "Security Operations Centre within 1 hour by calling extension 4400 or emailing soc@harbourline.example. "
                 "Do not try to investigate or clean an infected device yourself."),
                ("Data classification", "Information is classified as Public, Internal, Confidential or Restricted. Customer "
                 "shipment data is Confidential. Salary data and customs investigation files are Restricted and may only be "
                 "stored in approved systems."),
            ],
        ],
    },
    "travel_and_expense_policy.pdf": {
        "groups": ["all"],
        "title": "Travel and Expense Policy",
        "pages": [
            [
                ("Booking travel", "All business travel must be booked through the corporate travel portal at least 14 days "
                 "before departure where possible. Economy class is required for flights under 6 hours. Premium economy is "
                 "allowed for flights of 6 hours or more, and business class requires approval from a Vice President."),
                ("Hotels", "The nightly hotel cap is SGD 250 in Southeast Asia, SGD 350 in Europe and SGD 400 in North America, "
                 "excluding taxes. Stays above the cap need written approval from your manager before booking."),
            ],
            [
                ("Meals and per diem", "Meals while travelling are covered by a daily per diem of SGD 80, which does not require "
                 "receipts. Client entertainment is claimed separately with itemised receipts and the names of attendees."),
                ("Submitting claims", "Expense claims must be submitted in the finance portal within 30 days of the expense. "
                 "Receipts are required for every item above SGD 25. Claims are reimbursed in the next monthly payroll run "
                 "after approval."),
            ],
        ],
    },
    "compensation_guidelines.pdf": {
        "groups": ["hr"],
        "title": "Compensation Guidelines (Restricted)",
        "pages": [
            [
                ("Annual salary review", "Salary reviews take place every April. The merit budget for the current cycle is "
                 "4.5 percent of base payroll. Increases above 10 percent for any individual require approval from the Chief "
                 "People Officer."),
                ("Bonus", "The annual performance bonus target is one month of base salary for individual contributors and "
                 "two months for managers, adjusted by a company multiplier between 0 and 1.5."),
            ],
        ],
    },
    "incident_runbook.pdf": {
        "groups": ["engineering"],
        "title": "Platform Incident Runbook",
        "pages": [
            [
                ("Severity levels", "SEV1 means the shipment tracking platform is down for all customers, SEV2 means a major "
                 "feature is degraded, and SEV3 means a minor issue with a workaround. SEV1 incidents page the on call engineer "
                 "and the engineering manager immediately."),
                ("First response", "The incident commander opens a channel named inc followed by the date, posts a status "
                 "update every 30 minutes for SEV1 and every 60 minutes for SEV2, and records the timeline as events happen."),
            ],
            [
                ("Database failover", "If the primary tracking database is unreachable for more than 5 minutes, promote the "
                 "read replica in the secondary region using the failover pipeline. Confirm replication lag is below 30 "
                 "seconds before promoting to avoid data loss."),
                ("Postmortems", "Every SEV1 and SEV2 incident needs a blameless postmortem published within 5 working days, "
                 "with action items that have named owners and due dates."),
            ],
        ],
    },
}

TABLES = {
    "warehouse_sla.csv": {
        "groups": ["all"],
        "title": "Warehouse service levels by site",
        "rows": [
            ["site", "region", "dock_to_stock_hours", "order_cutoff", "same_day_dispatch", "on_time_target_pct"],
            ["Tuas", "Singapore", "24", "2pm", "yes", "98.5"],
            ["Changi Airfreight", "Singapore", "8", "4pm", "yes", "99.0"],
            ["Port Klang", "Malaysia", "36", "12pm", "no", "97.0"],
            ["Laem Chabang", "Thailand", "48", "11am", "no", "96.0"],
            ["Cikarang", "Indonesia", "48", "11am", "no", "95.5"],
        ],
    },
    "salary_bands.csv": {
        "groups": ["hr"],
        "title": "Salary bands (Restricted)",
        "rows": [
            ["level", "title", "band_min_sgd", "band_max_sgd"],
            ["L1", "Associate", "42000", "55000"],
            ["L2", "Specialist", "55000", "72000"],
            ["L3", "Senior Specialist", "72000", "95000"],
            ["L4", "Manager", "95000", "130000"],
        ],
    },
}


def write_pdf(path: Path, title: str, pages: list) -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    for page in pages:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 15)
        pdf.multi_cell(0, 9, title, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)
        for heading, body in page:
            pdf.set_font("Helvetica", "B", 12)
            pdf.multi_cell(0, 7, heading, new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(0, 6, body, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(3)
    pdf.output(str(path))


def main() -> None:
    ROOT.mkdir(exist_ok=True)
    manifest = {}
    for name, spec in DOCS.items():
        write_pdf(ROOT / name, spec["title"], spec["pages"])
        manifest[name] = {"groups": spec["groups"], "title": spec["title"]}
    for name, spec in TABLES.items():
        with (ROOT / name).open("w", newline="") as f:
            csv.writer(f).writerows(spec["rows"])
        manifest[name] = {"groups": spec["groups"], "title": spec["title"]}
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(manifest)} sources to {ROOT}")


if __name__ == "__main__":
    main()
