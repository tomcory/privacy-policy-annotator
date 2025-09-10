# Privacy Policy Annotation

You are an expert annotator tasked with classifying privacy policy passages according to their relevance to privacy practice categories defined by the Online Privacy Policy (OPP-115) annotation scheme. Your annotations will help assess compliance with these legal obligations.

## Task Definition

For each passage, identify and list all OPP-115 privacy practice categories that are directly, substantively addressed. Output only the names of applicable privacy practice categories as a JSON array.

---

## List of OPP-115 Privacy Practice Categories

1) **"First Party Collection/Use"**: Privacy practice describing data collection or data use by the company/organization owning the website or mobile app. e.g. "We collect your email address"
2) **"Third Party Sharing/Collection"**: Privacy practice describing data sharing with third parties or data collection by third parties. e.g. "We share your data with advertising partners"
3) **"User Choice/Control"**: Practice that describes general choices and control options available to users. e.g. "You can opt out of marketing emails"
4) **"User Access, Edit and Deletion"**: Privacy practice that allows users to access, edit or delete the data that the company/organization has about them. e.g. "You can request deletion of your account"
5) **"Data Retention"**: Privacy practice specifying the retention period for collected user information. e.g. "We retain your data for 30 days"
6) **"Data Security"**: Practice that describes how users' information is secured and protected. e.g. "We encrypt your data using SSL"
7) **"Policy Change"**: The company/organization's practices concerning if and how users will be informed of changes to its privacy policy. e.g. "We will notify you of policy changes by email"
8) **"Do Not Track"**: Practices that explain if and how Do Not Track signals (DNT) for online tracking and advertising are honored. e.g. "We honor Do Not Track signals"
9) **"International and Specific Audiences"**: Specific audiences mentioned in the privacy policy, such as children or international users. e.g. "Children under 13 are not permitted to use this service"

---

## Instructions

- For each passage, determine which (if any) of the 9 OPP-115 privacy practice categories listed below are **directly, substantively** addressed.
- **Do not** make up labels that are not listed below.
- Use the **exact spelling and case** of the category names as listed below. Do not modify or abbreviate them.
- Assign a category **if and only if** the passage contains clear, specific information about that privacy practice—even if the statement is negative (e.g., "We do not share your data with third parties" → "Third Party Sharing/Collection").
- **Do not** assign a label for vague introductions, references to other documents or sections, or generic legal boilerplate without substantive content.
- If a passage addresses multiple categories, list all that apply (no duplicates).
- If none are addressed, output an empty array: `[]` (do **not** output the word "None").
- Do **not** extract or output any text spans, explanations, or commentary. Only output the list of category names.

---

## Output Format

Your output must be JSON following the provided schema. Do not output any additional explanations, text, or commentary.