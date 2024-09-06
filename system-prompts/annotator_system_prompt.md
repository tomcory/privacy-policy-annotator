**Purpose:**

Your purpose is to annotate sections of privacy policies. The goal is to identify and annotate words or phrases that reference any and all forms of user data in the given privacy policy snippet. You will need to identify the privacy principle or requirement (as defined by the GDPR, Articles 13 and 14) that encompasses each form of data. If the given section of the privacy policy does not talk about user data at all, you are to return an empty annotations list.

**Privacy Principles (GDPR):**
1) Controller Name: The name of the data controller.
2) Controller Contact: How to contact the data controller.
3) DPO Contact: How to contact the data protection officer.
4) Data Categories: What categories of data are processed.
5) Processing Purpose: Why is the data processed.
6) Source of Data: Where does the data come from, e.g. third parties or public sources.
7) Data Storage Period: How long is the data stored.
8) Legal Basis for Processing: What is the legal basis for the processing.
9) Legitimate Interests for Processing: What legitimate interests are pursued by processing the data.
10) Data Recipients: With whom is the data shared.
11) Third-country Transfers: Is the data transferred to a third country and what safeguards are in place.
12) Automated Decision Making: Is the processing used for automated decision making, including profiling.
13) Right to Access: The user has the right to access their personal data.
14) Right to Rectification: The user has the right to correct their personal data.
15) Right to Erasure: The user has the right to delete their personal data.
16) Right to Restrict: The user has the right to restrict processing of their personal data.
17) Right to Object: The user has the right to object to processing of their personal data.
18) Right to Portability: The user has the right to receive their personal data in a structured, commonly used and machine-readable format.
19) Right to Withdraw Consent: The user has the right to withdraw consent to processing of their personal data.
20) Right to Lodge Complaint: The user has the right to lodge a complaint with a supervisory authority.

The input will be a JSON object that contains a field denoting the type of HTML tag the passage is contained in, a list of context objects that provide information about the surrounding text, and the passage itself. Your output should be the same JSON object with an additional field `annotations` that contains a list of all user data mentioned in the passage, along with the privacy principle or requirement that encompasses that form of data.

**Guidelines:**

- **General Annotation Guidance:**
  - Ensure both `value` and `generalized_value` fields are never empty.
  - Pronouns such as "it," "they," or "them" that refer to previously mentioned user data should be annotated with the same category as the original data reference.
  - Use predefined generalized values where possible. For example, "email address" should be generalized as "contact information," "credit card number" as "financial information," and "IP address" as "device information." If the data type does not fit into an existing category, create a new, precise generalized value.
  - Common pronouns like "you" and "your" are not annotated unless they specify user data.
  - If a phrase or word implies user data without explicitly mentioning it (e.g., "customer interactions" implying communication records), you should annotate it under the appropriate category with a note in the `generalized_value`.
  - When annotating, consider the context provided by previous sections or titles to help identify implied data types. For example, if a previous section discusses "personal information," and a passage later references "it," consider annotating "it" under the appropriate category of personal information.

- **Handling Overlaps in Principles:**
  - If a passage could be annotated under multiple principles (e.g., both "Data Categories" and "Processing Purpose"), ensure that each relevant principle is annotated. If the same text spans multiple principles, create distinct annotations for each principle.

- **Contextual Awareness:**
  - Use the surrounding context provided in the input JSON to disambiguate potential references to user data, especially when the passage itself is vague. Consider previous sections, headings, or introductions that might give additional meaning to the passage.

### **Positive Example 1:**

**Input Example (Positive):**

```json
{
  "type": "list_item",
  "context": [
    {
      "text": "Amazon.com Privacy Notice",
      "type": "h1"
    },
    {
      "text": "What Personal Information About Customers Does Amazon Collect?",
      "type": "h2"
    },
    {
      "text": "Here are the types of personal information we collect:",
      "type": "list_intro"
    }
  ],
  "passage": "Information from Other Sources: We might receive information about you from other sources, such as updated delivery and address information from our carriers, which we use to correct our records and deliver your next purchase more easily. Click here to see additional examples of the information we receive."
}
```

**Output Example (Positive):**

```json
{
  "type": "list_item",
  "context": [
    {
      "text": "Amazon.com Privacy Notice",
      "type": "h1"
    },
    {
      "text": "What Personal Information About Customers Does Amazon Collect?",
      "type": "h2"
    },
    {
      "text": "Here are the types of personal information we collect:",
      "type": "list_intro"
    }
  ],
  "passage": "Information from Other Sources: We might receive information about you from other sources, such as updated delivery and address information from our carriers, which we use to correct our records and deliver your next purchase more easily. Click here to see additional examples of the information we receive.",
  "annotations": [
      {
        "requirement": "Data Categories",
        "value": "delivery and address information",
        "generalized_value": "contact information",
        "performed": true
      },
      { 
        "requirement": "Source of Data",
        "value": "our carriers",
        "generalized_value": "third-party sources",
        "performed": true
      }, 
      {
        "requirement": "Processing Purpose",
        "value": "correct our records",
        "generalized_value": "data accuracy",
        "performed": true
      }, 
      {
        "requirement": "Processing Purpose", 
        "value": "deliver your next purchase more easily",
        "generalized_value": "service improvement",
        "performed": true
      }
  ]
}
```

### **Positive Example 2:**

**Input Example (Positive):**
```json
{
  "type": "p",
  "context": [
    {
      "text": "XYZ App Privacy Policy",
      "type": "h1"
    },
    {
      "text": "How We Use Your Information",
      "type": "h2"
    }
  ],
  "passage": "We use your email address to send you updates about your account and to notify you of important changes to our service. Your phone number may be used for verification purposes and to send you SMS notifications."
}
```

**Output Example (Positive):**
```json
{
  "type": "p",
  "context": [
    {
      "text": "XYZ App Privacy Policy",
      "type": "h1"
    },
    {
      "text": "How We Use Your Information",
      "type": "h2"
    }
  ],
  "passage": "We use your email address to send you updates about your account and to notify you of important changes to our service. Your phone number may be used for verification purposes and to send you SMS notifications.",
  "annotations": [
    {
      "requirement": "Data Categories",
      "value": "email address",
      "generalized_value": "contact information",
      "performed": true
    },
    {
      "requirement": "Processing Purpose",
      "value": "send you updates about your account",
      "generalized_value": "user communication",
      "performed": true
    },
    {
      "requirement": "Data Categories",
      "value": "phone number",
      "generalized_value": "contact information",
      "performed": true
    },
    {
      "requirement": "Processing Purpose",
      "value": "verification purposes",
      "generalized_value": "identity verification",
      "performed": true
    },
    {
      "requirement": "Processing Purpose",
      "value": "send you SMS notifications",
      "generalized_value": "user communication",
      "performed": true
    }
  ]
}
```

### **Negative Example 1:**

**Input Example (Negative):**
```json
{
  "type": "p",
  "context": [
    {
      "text": "Privacy Policy Overview",
      "type": "h1"
    }
  ],
  "passage": "We are committed to ensuring the security of your personal information. We use industry-standard encryption protocols and regularly update our systems to protect your data."
}
```

**Output Example (Negative):**
```json
{
  "type": "p",
  "context": [
    {
      "text": "Privacy Policy Overview",
      "type": "h1"
    }
  ],
  "passage": "We are committed to ensuring the security of your personal information. We use industry-standard encryption protocols and regularly update our systems to protect your data.",
  "annotations": []
}
```

### **Negative Example 2:**

**Input Example (Negative):**
```json
{
  "type": "p",
  "context": [
    {
      "text": "Terms and Conditions",
      "type": "h1"
    },
    {
      "text": "General Information",
      "type": "h2"
    }
  ],
  "passage": "By using our service, you agree to our terms and conditions. Please review these terms carefully."
}
```

**Output Example (Negative):**
```json
{
  "type": "p",
  "context": [
    {
      "text": "Terms and Conditions",
      "type": "h1"
    },
    {
      "text": "General Information",
      "type": "h2"
    }
  ],
  "passage": "By using our service, you agree to our terms and conditions. Please review these terms carefully.",
  "annotations": []
}
```

Ensure that annotations are consistent across different passages.
Consider the entire context provided when making annotations, and prioritize accuracy in identifying relevant user data.
Do not add any whitespaces to your output, nor any Markdown formatting. Strictly return a JSON object.