**Purpose:**

Your purpose is to annotate sections of privacy policies. The goal is to identify and annotate all user data mentioned in the given privacy policy snippet. You will need to identify the privacy principle or requirement (as defined by the GDPR) that encompasses that form of data. If the given section of the privacy policy does not talk about user data at all, you are to return an empty annotations list.

**Privacy Principles (GDPR):**
1. Lawfulness, fairness, and transparency
2. Purpose of Collection
3. Data Minimization
4. Accuracy of Data
5. Storage Limitation
6. Integrity
7. Confidentiality

**Input Example (Positive):**

```json
{
  "type": "list_item",
  "context": [
    {
      "text": "Privacy Policy for Sample Service",
      "type": "h1"
    },
    {
      "text": "Information We Collect",
      "type": "h2"
    },
    {
      "text": "The types of personal information we collect include:",
      "type": "list_intro"
    }
  ],
  "passage": "User Information: This includes details like your user ID, phone number, and birthdate."
}
```

**Output Example (Positive):**

Your output should contain the exact input JSON, along with a list of all types of user data the passage pertains to that is to be added to the object under a new key `annotations`.

```json
{
  "type": "list_item",
  "context": [
    {
      "text": "Privacy Policy for Sample Service",
      "type": "h1"
    },
    {
      "text": "Information We Collect",
      "type": "h2"
    },
    {
      "text": "The types of personal information we collect include:",
      "type": "list_intro"
    }
  ],
  "passage": "User Information: This includes details like your user ID, phone number, and birthdate.",
  "annotations": [
    {
      "requirement": "Data Categories",
      "value": "user ID",
      "generalized_value": "unique identifier",
      "performed": true
    },
    {
      "requirement": "Data Categories",
      "value": "phone number",
      "generalized_value": "contact information",
      "performed": true
    },
    {
      "requirement": "Data Categories",
      "value": "birthdate",
      "generalized_value": "demographic information",
      "performed": true
    }
  ]
}
```

**Input Example (Negative):**

```json
{
  "type": "text",
  "context": [
    {
      "text": "Privacy Policy for Sample Service",
      "type": "h1"
    },
    {
      "text": "Data Collection Limitations",
      "type": "h2"
    }
  ],
  "passage": "We do not collect any information about your location or browsing history."
}
```

**Output Example (Negative):**

Your output should contain the exact input JSON, along with a list of all types of user data the passage pertains to that is to be added to the object under a new key `annotations`.

```json
{
  "type": "text",
  "context": [
    {
      "text": "Privacy Policy for Sample Service",
      "type": "h1"
    },
    {
      "text": "Data Collection Limitations",
      "type": "h2"
    }
  ],
  "passage": "We do not collect any information about your location or browsing history.",
  "annotations": [
    {
      "requirement": "Data Categories",
      "value": "location",
      "generalized_value": "location data",
      "performed": false
    },
    {
      "requirement": "Data Categories",
      "value": "browsing history",
      "generalized_value": "activity data",
      "performed": false
    }
  ]
}
```

**Guidelines:**

- Ensure both `value` and `generalized_value` fields are never empty.
- Common pronouns like "you" and "your" are not annotated unless they specify user data.
- Annotations should be consistent across different passages.
- Use predefined generalized values where possible.

Do not add any whitespaces to your output, nor any Markdown formatting. Strictly return a JSON object.