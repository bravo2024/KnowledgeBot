from __future__ import annotations
"""Synthetic enterprise knowledge-base corpus for the KnowledgeBot RAG system.

A hand-authored collection of short policy / FAQ documents spanning six
departments (IT Policy, HR, Benefits, Security, Travel, Equipment).  Each
evaluation query is annotated with ground-truth relevant document ids so
retrieval quality can be measured with rank-aware metrics (Recall@k,
Precision@k, MRR, NDCG).

Reference
---------
Lewis et al. (2020), "Retrieval-Augmented Generation for Knowledge-Intensive
NLP Tasks", NeurIPS.
"""
from typing import Any


def _doc(doc_id: str, title: str, category: str, content: str) -> dict[str, Any]:
    return {"id": doc_id, "title": title, "category": category, "content": content}


def _q(qid: str, query: str, relevant: list[str], answer: str = "") -> dict[str, Any]:
    return {"id": qid, "query": query, "relevant_doc_ids": relevant, "reference_answer": answer}


DOCUMENTS: list[dict[str, Any]] = [
        _doc("KB-001", "Password Policy", "IT Policy", "Employees must use strong passwords of at least twelve characters combining uppercase, lowercase, numbers, and symbols. Passwords must be changed every ninety days and cannot reuse any of the last five passwords. Multi-factor authentication is required for all corporate accounts, and passwords must never be shared or stored in plain text."),
    _doc("KB-002", "VPN Remote Access", "IT Policy", "Employees working remotely must connect through the corporate VPN to access internal systems. The VPN client can be downloaded from the IT self-service portal using your single sign-on credentials. VPN sessions time out after eight hours of inactivity, after which you must reconnect."),
    _doc("KB-003", "Acceptable Use Policy", "IT Policy", "Company devices and networks are provided primarily for business purposes, though reasonable personal use is permitted. Installing unapproved software is prohibited and may be removed without notice. All network activity is monitored and logged in accordance with the acceptable use policy."),
    _doc("KB-004", "Data Classification", "IT Policy", "Company data is classified into four tiers: public, internal, confidential, and restricted. Confidential and restricted data must be encrypted both at rest and in transit. Access to restricted data requires manager approval and is logged, and confidential data must never be sent over personal email."),
    _doc("KB-005", "Software Installation", "IT Policy", "All software must be approved by IT and installed through the managed software catalog. Requests for new software should be submitted via the IT service portal with a business justification and manager approval. Unauthorized installations may be removed without notice and license compliance is reviewed quarterly."),
    _doc("KB-006", "Incident Reporting", "IT Policy", "Report all security incidents and suspicious emails to the security team immediately. Do not forward suspicious messages to colleagues or click any links inside them. Phishing attempts should be reported using the Report Phishing button in Outlook, and the security team responds within one business hour."),
    _doc("KB-007", "Device Encryption", "IT Policy", "All laptops and mobile devices must have full-disk encryption enabled. BitLocker is enabled by default on managed Windows devices and FileVault on managed Macs. Encryption status can be verified in the IT self-service portal, and lost or stolen devices must be reported to IT within one hour."),
        _doc("KB-008", "Paid Time Off", "HR", "Full-time employees accrue paid time off at fifteen days per year, increasing to twenty days after three years of service. PTO requests must be submitted at least two weeks in advance through the HR portal and approved by your manager. Up to five unused days may be carried over to the following calendar year."),
    _doc("KB-009", "Remote Work Policy", "HR", "Employees may work remotely up to three days per week with manager approval. Remote employees must be available during core hours of 10 AM to 3 PM local time and maintain a stable internet connection. Home-office equipment such as a monitor and keyboard can be requested through the IT portal."),
    _doc("KB-010", "Performance Reviews", "HR", "Performance reviews are conducted annually in December, with a mid-year check-in each June. Each review includes a self-assessment, peer feedback, and a manager evaluation against goals set using the SMART framework. Promotion decisions consider performance ratings, goal achievement, and business needs."),
    _doc("KB-011", "Onboarding", "HR", "New employees complete a five-day onboarding program covering company culture, tools, and policies. IT equipment is provisioned before day one and a buddy is assigned to each new hire for the first ninety days. Onboarding feedback is collected at thirty, sixty, and ninety days to improve the experience."),
    _doc("KB-012", "Code of Conduct", "HR", "Employees must maintain professional conduct and treat all colleagues with respect. Harassment and discrimination of any kind are strictly prohibited and must be reported to HR. Conflicts of interest must be disclosed promptly, and violations may result in disciplinary action up to and including termination."),
    _doc("KB-013", "Leave of Absence", "HR", "Employees may request a leave of absence for medical, personal, or family reasons. Medical leaves are covered under FMLA for up to twelve weeks with supporting documentation from a provider. Benefit continuation during a leave depends on the leave type and must be coordinated with HR before starting."),
    _doc("KB-014", "Timesheets", "HR", "All employees must submit weekly timesheets by 5 PM each Friday. Timesheets must accurately reflect hours worked against valid project codes for billing. Late or incomplete submissions may delay payroll processing, and corrections must be requested through the timekeeping system."),
    _doc("KB-015", "Health Insurance", "Benefits", "The company offers comprehensive health insurance through BlueCross BlueShield with PPO and HMO plan options. Coverage begins on the first day of the month following your hire date. Dependent coverage is available with a monthly premium contribution that varies by plan tier."),
    _doc("KB-016", "401k Retirement Plan", "Benefits", "Employees are eligible to enroll in the 401k retirement plan after thirty days of employment. The company matches one hundred percent of employee contributions up to five percent of base salary. Vesting of the employer match occurs after two years of service, and investment options include target-date and index funds."),
    _doc("KB-017", "Dental and Vision", "Benefits", "Dental and vision insurance are offered as voluntary benefits with premiums deducted pre-tax. The dental plan covers two routine cleanings per year at no additional cost. Vision coverage includes an annual eye exam and an allowance for frames or contact lenses."),
    _doc("KB-018", "Life Insurance", "Benefits", "The company provides basic life insurance equal to two times your annual salary at no cost. Supplemental life insurance up to five times salary is available for purchase, along with spouse and dependent coverage options. Beneficiaries can be updated at any time through the HR benefits portal."),
    _doc("KB-019", "Flexible Spending Account", "Benefits", "Employees may contribute pre-tax dollars to a health care or dependent care flexible spending account. The annual contribution limit is set by the IRS each plan year. Funds must be used within the plan year, with a grace period through March 15, and claims are submitted through the benefits portal."),
    _doc("KB-020", "Wellness Program", "Benefits", "The wellness program reimburses up to five hundred dollars annually for gym memberships and fitness equipment. Employees earn points by participating in wellness challenges and can redeem points for rewards. A free annual health screening is offered on site each fall."),
    _doc("KB-021", "Tuition Reimbursement", "Benefits", "Employees are eligible for up to five thousand dollars per year in tuition reimbursement for job-related courses. Courses must be approved by a manager before enrollment and a minimum grade of B is required for reimbursement. Applications and receipts are submitted through the HR learning portal."),
        _doc("KB-022", "Phishing Awareness", "Security", "Phishing emails attempt to steal credentials by impersonating trusted senders with urgent language and unexpected attachments. Watch for mismatched sender addresses and links that lead to unfamiliar websites. Never click suspicious links or enter credentials on sites you did not expect, and report phishing using the Report Phishing button."),
    _doc("KB-023", "Data Breach Response", "Security", "In the event of a data breach the security team activates the incident response plan within one hour. Affected systems are isolated immediately and forensic analysis begins to determine the scope. Employees must not discuss breaches externally, and the legal team handles any required regulatory notifications."),
    _doc("KB-024", "Access Management", "Security", "Access to systems is granted based on role using least-privilege principles. Access requests require manager approval and all entitlements are reviewed quarterly. Shared accounts are prohibited, and any unnecessary access should be reported to the IT helpdesk for removal."),
    _doc("KB-025", "Mobile Device Security", "Security", "Mobile devices that access company data must enroll in the mobile device management system. A six-digit PIN and biometric lock are required, and remote wipe is enabled for lost devices. Only approved applications may access corporate email and data."),
    _doc("KB-026", "Secure File Sharing", "Security", "Confidential files must be shared using the approved enterprise file-sharing platform. External sharing links expire after seven days and should be password-protected for sensitive content. Personal file-sharing services must never be used to share company data."),
    _doc("KB-027", "Security Awareness Training", "Security", "All employees must complete annual security awareness training within the first thirty days of assignment. Additional role-based training is assigned depending on job function and access level. Training completion is tracked and reported to managers, and new hires complete it during onboarding."),
    _doc("KB-028", "Two-Factor Authentication", "Security", "Two-factor authentication is required for all external access to corporate systems. Use the approved authenticator app as your primary method and store backup codes in a secure location. Contact IT immediately if you lose access to your authentication device."),
    _doc("KB-029", "Travel Booking", "Travel", "All business travel must be booked through the corporate travel portal to ensure policy compliance. Book flights at least fourteen days in advance when possible and use economy class for flights under five hours. International travel requires vice-president approval before booking."),
    _doc("KB-030", "Expense Reimbursement", "Travel", "Submit expense reports within thirty days of travel through the expense management system. Original receipts are required for all expenses over twenty-five dollars and must be attached to the report. Reimbursement is processed within ten business days of manager approval."),
    _doc("KB-031", "Travel Insurance", "Travel", "The company provides travel insurance for all business trips booked through the portal. Coverage includes trip cancellation, lost baggage, and medical emergencies while traveling. Keep all receipts and documentation to support any claims, and contact the travel desk during emergencies."),
    _doc("KB-032", "International Travel", "Travel", "International travel requires approval at least four weeks in advance. A valid passport with six months of remaining validity is required, and visas are arranged through the corporate travel agency. Register your itinerary with the security team so travel risk can be monitored."),
    _doc("KB-033", "Ground Transportation", "Travel", "Employees may use rental cars, taxis, or rideshare services for business travel. Rental cars must be booked through the travel portal with company insurance and a valid license. Personal vehicle mileage is reimbursed at the current IRS standard rate with receipts for tolls and parking."),
    _doc("KB-034", "Hotel Booking", "Travel", "Book hotels through the corporate travel portal to ensure compliance and access preferred rates. The nightly rate should not exceed the per diem for the destination city. Loyalty program numbers may be used for personal benefit, and cancellation policies vary by property."),
    _doc("KB-035", "Travel Per Diem", "Travel", "The daily per diem covers meals and incidentals and varies by city as listed in the travel policy. The per diem is paid as a flat rate and no receipts are required for meals within the limit. Expenses exceeding the per diem require additional manager approval and documentation."),
        _doc("KB-036", "Laptop Provisioning", "Equipment", "Standard laptops are provisioned for new employees before their start date. Replacement requests for damaged or outdated laptops are submitted via the IT portal. Loaner laptops are available during repairs and all laptops must be returned upon termination of employment."),
    _doc("KB-037", "Monitor and Accessories", "Equipment", "Employees may request one additional monitor and standard accessories through the IT portal. Ergonomic equipment such as a standing desk requires approval through the wellness program. Requests are fulfilled within five business days and remote employees receive equipment shipped to their home."),
    _doc("KB-038", "Phone Provisioning", "Equipment", "Company phones are provided to employees whose roles require mobile access. You may choose between iOS and Android devices, and international roaming must be enabled before travel. A personal phone stipend is available for eligible roles under the bring-your-own-device program."),
    _doc("KB-039", "Printer Setup", "Equipment", "Network printers are available on every floor and configured automatically through the IT portal. Color printing requires a valid cost-center code entered at the device. Report paper jams and malfunctions to facilities and do not attempt to service equipment yourself."),
    _doc("KB-040", "Conference Room AV", "Equipment", "Conference rooms are equipped with video conferencing systems compatible with Teams and Zoom. Reserve rooms through the room booking system and start meetings on time. Technical support is available via the AV help button in each room, and equipment issues should be reported to IT."),
    _doc("KB-041", "Equipment Return", "Equipment", "All company equipment must be returned on your last day of employment. This includes laptops, phones, badges, monitors, and accessories. IT inspects and deactivates returned devices, and failure to return equipment may result in deductions from your final paycheck."),
    _doc("KB-042", "Ergonomic Assessment", "Equipment", "Employees may request an ergonomic workstation assessment through the wellness program. An ergonomics specialist evaluates your desk setup and posture and may recommend a standing desk or ergonomic chair. Approved equipment is provided at no cost and setup is handled by facilities.")
]

QUERIES: list[dict[str, Any]] = [
        _q("Q01", "What is the password policy and how long must passwords be?", ["KB-001"], "Passwords must be at least twelve characters and changed every ninety days."),
    _q("Q02", "How do I connect to the corporate VPN when working remotely?", ["KB-002"], "Use the VPN client from the IT self-service portal to access internal systems."),
    _q("Q03", "How many days of paid time off do full-time employees receive?", ["KB-008"], "Fifteen days per year, increasing to twenty after three years."),
    _q("Q04", "What is the company match for the 401k retirement plan?", ["KB-016"], "The company matches contributions up to five percent of base salary."),
    _q("Q05", "How do I report a phishing email I received?", ["KB-006", "KB-022"], "Use the Report Phishing button in Outlook and do not click any links."),
    _q("Q06", "How do I request a new or replacement laptop?", ["KB-036"], "Submit a replacement request via the IT portal; loaners are available during repairs."),
    _q("Q07", "What is the remote work policy and how many days can I work from home?", ["KB-009"], "Up to three days per week with manager approval."),
    _q("Q08", "How do I book a business flight for company travel?", ["KB-029"], "Book through the corporate travel portal at least fourteen days in advance."),
    _q("Q09", "How do I submit an expense report for travel reimbursement?", ["KB-030"], "Submit within thirty days through the expense management system with receipts."),
    _q("Q10", "What health insurance plans are offered to employees?", ["KB-015"], "PPO and HMO plans through BlueCross BlueShield."),
    _q("Q11", "How do I set up two-factor authentication for external access?", ["KB-028"], "Use the approved authenticator app and store backup codes securely."),
    _q("Q12", "What is the tuition reimbursement limit per year?", ["KB-021"], "Up to five thousand dollars per year for job-related courses."),
    _q("Q13", "How is company data classified and how should confidential data be handled?", ["KB-004"], "Data is public, internal, confidential, or restricted; confidential data must be encrypted."),
    _q("Q14", "How do I install new software on my company laptop?", ["KB-005"], "Use the managed software catalog and submit requests via the IT service portal."),
    _q("Q15", "When are performance reviews conducted during the year?", ["KB-010"], "Annually in December with a mid-year check-in in June."),
    _q("Q16", "How do I share confidential files securely with external parties?", ["KB-026"], "Use the approved enterprise file-sharing platform with expiring password-protected links."),
    _q("Q17", "What are the security awareness training requirements?", ["KB-027"], "Annual training must be completed within thirty days of assignment."),
    _q("Q18", "How do I request an ergonomic assessment of my workstation?", ["KB-042"], "Request an assessment through the wellness program.")
]


def make_corpus() -> dict[str, Any]:
    """Return the knowledge-base corpus and the labelled evaluation queries."""
    categories = sorted({d["category"] for d in DOCUMENTS})
    return {
        "documents": [dict(d) for d in DOCUMENTS],
        "queries": [dict(q) for q in QUERIES],
        "categories": categories,
        "n_documents": len(DOCUMENTS),
        "n_queries": len(QUERIES),
    }
