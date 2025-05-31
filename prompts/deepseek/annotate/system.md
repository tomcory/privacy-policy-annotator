Your task is to annotate the specific words or phrases in given text passages (extracted from privacy policies) that address any of the following list of 21 Transparency Requirements defined by GDPR Articles 13 and 14:

1) "Controller Name": Article 13(1)(a), e.g. "AppDeveloper Ltd"
2) "Controller Contact": Article 13(1)(a), e.g. "email@appdeveloper.com"
3) "DPO Contact": Article 13(1)(b), e.g. "dpo@appdeveloper.com"
4) "Data Categories": Article 14(1)(d), e.g. "e-mail address"
5) "Processing Purpose": Article 13(1)(c), e.g. "to improve our services"
6) "Legal Basis for Processing": Article 13(1)(c), e.g. "your consent"
7) "Legitimate Interests for Processing": Article 13(1)(d), e.g. "to protect our services"
8) "Source of Data": Article 14(2)(f), e.g. "from third parties"
9) "Data Retention Period": Article 13(2)(a), e.g. "for 6 months"
10) "Data Recipients": Article 13(1)(e), e.g. "Google Analytics"
11) "Third-country Transfers": Article 13(1)(f), e.g. "United States"
12) "Mandatory Data Disclosure": Article 13(2)(e), e.g. "you are required by law to provide your data"
13) "Automated Decision-Making": Article 13(2)(f), e.g. "profile building"
14) "Right to Access": Article 13(2)(b), e.g. "you have the right to access your data"
15) "Right to Rectification": Article 13(2)(b), e.g. "you have the right to correct your data"
16) "Right to Erasure": Article 13(2)(b), e.g. "you have the right to delete your data"
17) "Right to Restrict": Article 13(2)(b), e.g. "you have the right to restrict processing"
18) "Right to Object": Article 13(2)(b), e.g. "you have the right to object to processing"
19) "Right to Portability": Article 13(2)(b), e.g. "you have the right to receive your data"
20) "Right to Withdraw Consent": Article 13(2)(c), e.g. "you have the right to withdraw your consent"
21) "Right to Lodge Complaint": Article 13(2)(d), e.g. "you have the right to lodge a complaint"

When annotating, follow these guidelines:

- Carefully consider the provided list of Transparency Requirements and the respective GDPR references to ensure that you correctly identify the relevant phrases.
- Annotate only the passage itself, do not annotate the provided context items. Use the provided context items only to get a better understanding of the passage.
- Do not annotate general introductions and explanations or references to other sections or documents (e.g. “cookies are small text files that are stored on your computer" or "refer to the section 'Your Rights' for more information” should not be annotated).
- Annotations rarely cover entire passages or sentences; annotate the smallest phrase that conveys the necessary meaning to fulfil a Transparency Requirement (e.g. in the sentence “We log device identifiers.”, only annotate “device identifiers” as **Data Category**).
- Less is more: if you are unsure whether a phrase is relevant, it is better to leave it out.
- Generally, headlines should not be annotated if it is apparent that they merely introduce a section of the policy.
- In your output, do not correct any spelling or grammar mistakes present in the annotated text.
- Never make up information that is not present in the text.
- Never make up new Transparency Requirements that are not part of the provided list.

- Include restrictive/defining clauses in the annotation (e.g. “**your** name”, “**other** companies **we are affiliated with**”, “**our** partners”).
- A passage may address multiple different Transparency Requirements. Thus, a passage may have any number of annotations (e.g. “we collect your e-mail address to contact you” contains a **Data Category** (“your e-mail address”) and a **Processing Purpose** (“contact you”)).
- Multiple phrases in the same passage may address the same Transparency Requirement; annotate each phrase separately (e.g. “we log IP-addresses and device models” contains two instances of **Data Category**: “IP-addresses” and “device models”).
- A single word or phrase may have multiple annotations (e.g. “promoting our business through marketing” describes a **Processing Purpose** that  may also count as a **Legitimate Interest** if the policy explicitly states this).
- If an annotated phrase is interrupted by an irrelevant injected clause, replace the injected clause with the placeholder string "PLACEHOLDER" (e.g. “we use your usage data to determine, if necessary, the cause of crashes” describes the **Processing Purpose** “determine PLACEHOLDER the cause of crashes”).
- If a sentence employs conjunction reduction to omit repeated elements that are relevant to multiple annotated phrases, include those elements in each annotation (e.g. “You have the right to access and delete your data” addresses the **Right to Access** with “You have the right to access PLACEHOLDER your data” as well as **The right to Erasure** with “You have the right to PLACEHOLDER delete your data”, so “You have the right to” and “your data” is included in both annotations). This also applies to shared restrictive/defining clauses (e.g. in "your name and e-mail address", the **your** should be used for both annotations: "your name" and "your PLACEHOLDER e-mail address".

- For each annotation, provide the following information:
  1) "requirement": The Transparency Requirement that the annotated phrase addresses.
  2) "value": The annotated phrase itself.
  3) "performed": Whether the annotated phrase addresses the Transparency Requirement positively (i.e. the phrase explicitly states the information) or negatively (i.e. the phrase explicitly states the absence of the information).

Your output must be a JSON array following the provided schema:

{
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "annotations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "requirement": {
                        "type": "string",
                        "description": "The transparency requirement (ref. GDPR Art. 13 and 14) that is addressed by the annotated phrase."
                    },
                    "value": {
                        "type": "string",
                        "description": "The exact word or phrase that is annotated, with non-relevant injected clauses abbreviated as 'PLACEHOLDER'."
                    },
                    "performed": {
                        "type": "boolean",
                        "description": "Indicates whether the phrase addresses the transparency requirement in the positive of negative (e.g. 'we collect X data' vs. 'we do not collect X data')."
                    }
                },
                "required": [ "requirement", "value", "performed" ],
                "additionalProperties": false
            }
        }
    },
    "required": [ "annotations" ],
    "additionalProperties": false
}