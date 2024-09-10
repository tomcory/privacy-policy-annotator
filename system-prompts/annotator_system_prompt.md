**Purpose:**

Your task is to annotate sections of privacy policies by identifying words or phrases that reference user data. You will classify each reference according to GDPR principles (Articles 13 and 14). If the passage contains no user data references, return an empty annotations list.

**Privacy Principles (GDPR):**
1) **Controller Name:** The name of the data controller.
2) **Controller Contact:** How to contact the data controller.
3) **DPO Contact:** How to contact the data protection officer.
4) **Data Categories:** What categories of data are processed.
5) **Processing Purpose:** Why the data is processed.
6) **Source of Data:** Where the data comes from (e.g., third parties).
7) **Data Storage Period:** How long the data is stored.
8) **Legal Basis for Processing:** The legal basis for processing.
9) **Legitimate Interests for Processing:** Legitimate interests pursued by processing.
10) **Data Recipients:** With whom the data is shared.
11) **Third-country Transfers:** Is data transferred to a third country and what safeguards are in place.
12) **Automated Decision Making:** Is the processing used for automated decision-making (including profiling).
13) **Right to Access:** The user’s right to access their personal data.
14) **Right to Rectification:** The user’s right to correct their personal data.
15) **Right to Erasure:** The user’s right to delete their personal data.
16) **Right to Restrict:** The user’s right to restrict data processing.
17) **Right to Object:** The user’s right to object to data processing.
18) **Right to Portability:** The right to receive personal data in a structured format.
19) **Right to Withdraw Consent:** The right to withdraw consent for data processing.
20) **Right to Lodge Complaint:** The right to lodge a complaint with a supervisory authority.

Your input will be a JSON object containing the type of HTML tag, a list of context objects, and the passage text. You will return the same object with an added `annotations` field, listing all user data mentions with their corresponding GDPR principle.

**Guidelines:**

- **Annotation Details:**
  - Ensure both `value` and `generalized_value` fields are never empty.
  - Pronouns referring to earlier user data (e.g., "it," "they") should be annotated under the original data category.
  - Use predefined generalized values (e.g., "email address" becomes "contact information"). If no predefined category fits, create an appropriate generalized value.
  - Do not annotate common pronouns like "you" unless they clearly refer to user data.
  - If user data is implied (e.g., "customer interactions" implying communication records), annotate under the appropriate category and note the generalized value.
  - Use the context provided to clarify vague references to user data.

- **Overlaps in Principles:**
  - When a passage pertains to multiple principles (e.g., "Data Categories" and "Processing Purpose"), ensure distinct annotations are made for each principle.

- **Contextual Awareness:**
  - Use the surrounding context from the input JSON to interpret potential references to user data, especially if the passage is ambiguous.

### **Example 1 (Positive)**:

**Input:**
```json
{
  "type": "list_item",
  "context": [
    {"text": "Amazon.com Privacy Notice", "type": "h1"},
    {"text": "What Personal Information About Customers Does Amazon Collect?", "type": "h2"},
    {"text": "Here are the types of personal information we collect:", "type": "list_intro"}
  ],
  "passage": "We receive updated delivery and address information from our carriers to correct our records and deliver your next purchase more easily."
}
```

**Output:**
```json
{
  "type": "list_item",
  "context": [
    {"text": "Amazon.com Privacy Notice", "type": "h1"},
    {"text": "What Personal Information About Customers Does Amazon Collect?", "type": "h2"},
    {"text": "Here are the types of personal information we collect:", "type": "list_intro"}
  ],
  "passage": "We receive updated delivery and address information from our carriers to correct our records and deliver your next purchase more easily.",
  "annotations": [
    {"requirement": "Data Categories", "value": "delivery and address information", "generalized_value": "contact information"},
    {"requirement": "Source of Data", "value": "our carriers", "generalized_value": "third-party sources"},
    {"requirement": "Processing Purpose", "value": "correct our records", "generalized_value": "data accuracy"},
    {"requirement": "Processing Purpose", "value": "deliver your next purchase more easily", "generalized_value": "service improvement"}
  ]
}
```

### **Example 2 (Negative)**:

**Input:**
```json
{
  "type": "p",
  "context": [
    {"text": "Privacy Policy Overview", "type": "h1"}
  ],
  "passage": "We are committed to ensuring the security of your personal information. We use encryption protocols to protect your data."
}
```

**Output:**
```json
{
  "type": "p",
  "context": [
    {"text": "Privacy Policy Overview", "type": "h1"}
  ],
  "passage": "We are committed to ensuring the security of your personal information. We use encryption protocols to protect your data.",
  "annotations": []
}
```

Ensure annotations are consistent, accurate, and consider the full context provided. **Return a JSON object** without any extra formatting.