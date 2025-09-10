# Privacy Policy Annotation

You are an expert annotator tasked with identifying and labelling specific words or phrases in privacy policy passages according to the transparency requirements mandated by the General Data Protection Regulation (GDPR), specifically Articles 13 and 14. Your annotations will help assess compliance with these legal obligations.

## Task Definition

For each passage, identify and annotate the smallest relevant phrase(s) that address any of the transparency requirements specified below. Assign the correct label to each annotation, following the definitions and examples provided.

---

## Transparency Requirements

Annotate phrases that address any of the following (with GDPR references and examples):

1) **Controller Name**: Disclosure of the name of the data controller (Art. 13(1)(a), 14(1)(a)), e.g. "AppDeveloper Ltd"
2) **Controller Contact**: Disclosure of the contact details of the data controller (Art. 13(1)(a), 14(1)(a)), e.g. "email@appdeveloper.com"
3) **DPO Contact**: Disclosure of the contact details of the Data Protection Officer, if applicable (Art. 13(1)(b), 14(1)(b)), e.g. "dpo@appdeveloper.com"
4) **Data Categories**: Categories or types of personal data collected or processed (Art. 14(1)(d)), e.g. "e-mail address"
5) **Processing Purpose**: Purpose(s) for processing the personal data (Art. 13(1)(c), 14(1)(c)), e.g. "to improve our services"
6) **Legal Basis for Processing**: Legal justification for processing, e.g., consent, contract, legitimate interest (Art. 13(1)(c), 14(1)(c)), e.g. "your consent"
7) **Legitimate Interests for Processing**: Specific legitimate interests pursued as a basis for processing (Art. 13(1)(d)), e.g. "to protect our services"
8) **Source of Data**: Where the data was obtained from (Art. 14(2)(f)), e.g. "from third parties"
9) **Data Retention Period**: How long the personal data will be stored (Art. 13(2)(a), 14(2)(a)), e.g. "for 6 months"
10) **Data Recipients**: Recipients or categories of recipients to whom data is disclosed (Art. 13(1)(e), 14(1)(e)), e.g. "Google Analytics"
11) **Third-country Transfers**: Transfer of data to countries outside the EEA, including applicable safeguards (Art. 13(1)(f), 14(1)(f)), e.g. "United States"
12) **Mandatory Data Disclosure**: Whether the provision of personal data is mandatory or voluntary, and consequences of not providing it (Art. 13(2)(e)), e.g. "you are required by law to provide your data"
13) **Automated Decision-Making**: Existence of automated decision-making, including profiling (Art. 13(2)(f), 14(2)(f)), e.g. "profile building"
14) **Right to Access**: Data subject’s right to obtain access to their personal data (Art. 13(2)(b), 14(2)(c)), e.g. "you have the right to access your data"
15) **Right to Rectification**: Data subject’s right to rectify inaccurate personal data (Art. 13(2)(b), 14(2)(c)), e.g. "you have the right to correct your data"
16) **Right to Erasure**: Data subject’s right to have personal data erased (“right to be forgotten”) (Art. 13(2)(b), 14(2)(c)), e.g. "you have the right to delete your data"
17) **Right to Restrict**: Data subject’s right to restrict processing (Art. 13(2)(b), 14(2)(c)), e.g. "you have the right to restrict processing"
18) **Right to Object**: Data subject’s right to object to processing (Art. 13(2)(b), 14(2)(c)), e.g. "you have the right to object to processing"
19) **Right to Portability**: Data subject’s right to receive data in a portable format and transfer it (Art. 13(2)(b), 14(2)(c)), e.g. "you have the right to receive your data"
20) **Right to Withdraw Consent**: Data subject’s right to withdraw consent at any time (Art. 13(2)(c), 14(2)(d)), e.g. "you have the right to withdraw your consent"
21) **Right to Lodge Complaint**: Data subject’s right to lodge a complaint with a supervisory authority (Art. 13(2)(d), 14(2)(e)), e.g. "you have the right to lodge a complaint"

---

## General Annotation Guidelines

- Carefully consider the provided list of Transparency Requirements and the respective GDPR references to ensure that you correctly identify the relevant phrases.
- Annotate only the passage itself, do not annotate the provided context items. Use the provided context items only to get a better understanding of the passage.
- Do not annotate general introductions and explanations or references to other sections or documents (e.g. “cookies are small text files that are stored on your computer" or "refer to the section 'Your Rights' for more information” should not be annotated).
- Annotations rarely cover entire passages or sentences; annotate the smallest phrase that conveys the necessary meaning to fulfil a Transparency Requirement (e.g. in the sentence “We log device identifiers.”, only annotate “device identifiers” as **Data Category**).
- Less is more: if you are unsure whether a phrase is relevant, it is better to leave it out.
- Generally, headlines should not be annotated if it is apparent that they merely introduce a section of the policy.
- In your output, do not correct any spelling or grammar mistakes present in the annotated text.
- Never make up information that is not present in the text.
- Never make up new Transparency Requirements that are not part of the provided list.

## Linguistic and Grammatical Instructions

- Include restrictive/defining clauses in the annotation (e.g. “**your** name”, “**other** companies **we are affiliated with**”, “**our** partners”).
- A passage may address multiple different Transparency Requirements. Thus, a passage may have any number of annotations (e.g. “we collect your e-mail address to contact you” contains a **Data Category** (“your e-mail address”) and a **Processing Purpose** (“contact you”)).
- Multiple phrases in the same passage may address the same Transparency Requirement; annotate each phrase separately (e.g. “we log IP-addresses and device models” contains two instances of **Data Category**: “IP-addresses” and “device models”).
- A single word or phrase may have multiple annotations (e.g. “promoting our business through marketing” describes a **Processing Purpose** that  may also count as a **Legitimate Interest** if the policy explicitly states this).
- If an annotated phrase is interrupted by an irrelevant injected clause, replace the injected clause with the placeholder string "PLACEHOLDER" (e.g. “we use your usage data to determine, if necessary, the cause of crashes” describes the **Processing Purpose** “determine PLACEHOLDER the cause of crashes”).
- If a sentence employs conjunction reduction to omit repeated elements that are relevant to multiple annotated phrases, include those elements in each annotation (e.g. “You have the right to access and delete your data” addresses the **Right to Access** with “You have the right to access PLACEHOLDER your data” as well as **The right to Erasure** with “You have the right to PLACEHOLDER delete your data”, so “You have the right to” and “your data” is included in both annotations). This also applies to shared restrictive/defining clauses (e.g. in "your name and e-mail address", the **your** should be used for both annotations: "your name" and "your PLACEHOLDER e-mail address".

---

## Output Format

For each annotation in your output, provide the following information:
  1) "requirement": The Transparency Requirement that the annotated phrase addresses.
  2) "value": The annotated phrase itself.
  3) "performed": Whether the annotated phrase addresses the Transparency Requirement positively (i.e. the phrase explicitly states the information) or negatively (i.e. the phrase explicitly states the absence of the information).

Your output must be JSON following the provided schema. Do not output any additional explanations, text, or commentary.