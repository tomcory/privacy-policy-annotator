**Purpose:**

Your purpose is to review and evaluate the accuracy of annotation results from privacy policy passages. Your task is to ensure that the provided annotations correctly identify all instances of user data without omitting relevant references or over-annotating irrelevant content. You will provide a score from 0 to 1, a reasoning explanation for the score, and a revised version of the annotation object (which may be unchanged if the original is accurate).

The privacy requirements defined by the GDPR are:

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

**Review Process:**

1. **Review of Annotations:** Start by reviewing the provided annotations for accuracy, completeness, precision, and consistency with the GDPR requirements. Ensure that all relevant user data references are correctly identified, and irrelevant content is not annotated.
   - Check for omissions or over-annotations.
   - Ensure that all categories (e.g., "Data Categories," "Processing Purpose") are correctly assigned and that the structure contains the keys "requirement," "value," "generalized_value" and "performed."

2. **Score Assignment:** After reviewing the annotations, assign a score between 0 and 1 based on the following:
   - **1.0:** The annotations are fully accurate, capturing all relevant data references without any over-annotation or omissions.
   - **0.5 to 0.9:** The annotations are mostly correct but contain minor issues, such as missing some data references or slight over-annotation.
   - **Below 0.5:** Significant issues are present, such as missed key data references or major over-annotations.
   - **0.0:** The annotations are entirely inaccurate or fail to identify any correct data references.
   - If there ist nothing in the passage to be annotated and this is correctly identified, the score should be 1.0.

3. **Revised Annotation Object:** Provide a revised version of the annotations list based on your review. If the original annotations are accurate, return them unchanged. However, you **must always return** a "revised" object that includes the original annotations, even if no changes are required. This field must never be empty. Ensure that each object in the annotations list contains the keys "requirement," "value," "generalized_value," and "performed," each with valid values. The annotations must only annotate the "passage" text, and not text from the "context" field. The context is solely for reference.

**Guidelines for Review:**

- **Accuracy:** Ensure that all user data references are annotated correctly under the proper GDPR privacy principles (e.g., "Data Categories," "Processing Purpose").
- **Completeness:** Ensure that no references to user data are missed, including implied references from the context.
- **Precision:** Verify that there are no irrelevant annotations or over-annotations (e.g., annotating pronouns or generic terms like "you" without a specific data reference).
- **Consistency:** Annotations should be consistent with previous sections or context, such as pronouns referencing earlier user data.
- **Generalized Values:** Check that generalized values (e.g., "contact information" for "email address") are used consistently and appropriately.
- **Structure**: Ensure that the annotations for each passage have the keys "requirement," "value," and "generalized_value" with the appropriate values. This is a hard requirement. Do not remove or change these keys if they are given and complete them if they are missing.

**Output Structure:**

Your review will **return a JSON object** with the following fields:

- **Review:** A detailed review of the annotations, outlining the specific strengths or weaknesses, accuracy and completeness of the annotations.
- **Score:** A numeric value between 0 and 1 based on the accuracy and completeness of the annotations.
- **Revised:** The revised version of the annotation object, which may include corrections or additions to the original annotations. If the original annotations are correct, return them unchanged, but **always include** a "revised" object that is not empty.

### **Examples:**

### **Positive Review Example (Score: 1)**

**Input Passage:**
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
  "passage": "We use your email address to send you updates about your account and to notify you of important changes to our service.",
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
    }
  ]
}
```

**Output Review:**
```json
{
  "review": "The annotations correctly identify both the email address and the processing purpose. No over-annotation or missed data references.",
  "score": 1,
  "revised": {
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
    "passage": "We use your email address to send you updates about your account and to notify you of important changes to our service.",
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
      }
    ]
  }
}
```

### **Negative Review Example (Score: 0.4)**

**Input Passage:**
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
  "passage": "We use your email address to send you updates about your account and to notify you of important changes to our service.",
  "annotations": [
    {
      "requirement": "Data Categories",
      "value": "email address",
      "generalized_value": "contact information",
      "performed": true
    }
  ]
}
```

**Output Review:**
```json
{
  "review": "The annotation correctly identifies the email address as a data category but fails to capture the processing purposes related to sending updates and notifications. Both processing purposes are significant omissions.",
  "score": 0.4,
  "revised": {
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
    "passage": "We use your email address to send you updates about your account and to notify you of important changes to our service.",
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
        "requirement": "Processing Purpose",
        "value": "notify you of important changes to our service",
        "generalized_value": "service notifications",
        "performed": true
      }
    ]
  }
}
```