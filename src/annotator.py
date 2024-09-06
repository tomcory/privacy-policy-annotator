import json

from src import util, api_wrapper

system_msg = '''Your task is to annotate the specific words or phrases in given text passages (extracted from privacy 
policies) that fulfill any of the following transparency requirements as defined by GDPR Articles 13 and 14: """ 1) 
Controller Name: The name of the data controller. 2) Controller Contact: How to contact the data controller. 3) DPO 
Contact: How to contact the data protection officer. 4) Data Categories: What categories of data are processed. 5) 
Processing Purpose: Why is the data processed. 6) Source of Data: Where does the data come from, e.g. third parties 
or public sources. 7) Data Storage Period: How long is the data stored. 8) Legal Basis for Processing: What is the 
legal basis for the processing. (Valid legal bases are consent, contractual necessity, legal obligation, protection 
of vital interests, public interest, and legitimate interests.) 9) Legitimate Interests for Processing: What 
legitimate interests are pursued by processing the data. (This annotation should only be used if the text 
**explicitly** declares that a given phrase describes a legitimate interest.) 10) Data Recipients: With whom is the 
data shared. 11) Third-country Transfers: Is the data transferred to a third country and what safeguards are in 
place. 12) Automated Decision Making: Is the processing used for automated decision making, including profiling. 13) 
Right to Access: The user has the right to access their personal data. 14) Right to Rectification: The user has the 
right to correct their personal data. 15) Right to Erasure: The user has the right to delete their personal data. 16) 
Right to Restrict: The user has the right to restrict processing of their personal data. 17) Right to Object: The 
user has the right to object to processing of their personal data. 18) Right to Portability: The user has the right 
to receive their personal data in a structured, commonly used and machine-readable format. 19) Right to Withdraw 
Consent: The user has the right to withdraw consent to processing of their personal data. 20) Right to Lodge 
Complaint: The user has the right to lodge a complaint with the controller or a supervisory authority. """ - Annotate 
only the passage itself, not the context items. Use the context items for clarification purposes, especially for list 
items or table cells. If a Transparency Requirement is addressed by a context item but not the passage, do not add an 
annotation for it. - Do not annotate general explanations or references to other sections or documents (e.g. “cookies 
are small text files that are stored on your computer" or "refer to the section 'Your Rights' for more information”). 
- Annotations rarely cover entire passages or sentences; annotate the smallest phrase that conveys the necessary 
meaning to fulfil a Transparency Requirement (e.g. in the sentence “We log device identifiers.”, only annotate 
“device identifiers” as **Data Category**). - Include restrictive/defining clauses in the annotation (e.g. “**your** 
name”, “**other** companies **we are affiliated with**”, “**our** partners”). - A passage may address multiple 
different Transparency Requirements. Thus, a passage may have any number of annotations (e.g. “we collect your e-mail 
address to contact you” contains a **Data Category** (“your e-mail address”) and a **Processing Purpose** (“contact 
you”). - A Transparency Requirement may be addressed by multiple phrases in the same passage; annotate each phrase 
separately (e.g. “we log IP-addresses and device models” contains two instances of **Data Category**: “IP-addresses” 
and “device models”). - A single word or phrase may have multiple annotations (e.g. “promoting our business through 
marketing” describes a **Processing Purpose** that also counts as a **Legitimate Interest** if the policy explicitly 
states this). - If an annotated phrase is interrupted by a non-relevant injected clause, wrap the injected clause in 
double square brackets (e.g. “we use your usage data to determine, if necessary, the cause of crashes” describes the 
**Processing Purpose** “determine [[, if necessary,]] the cause of crashes”). - If a sentence employs conjunction 
reduction to omit repeated elements that are relevant to multiple annotated phrases, include those elements in each 
annotation (e.g. “You have the right to access and delete your data” addresses the **Right to Access** with “You have 
the right to access [[and delete]] your data” as well as **The right to Erasure** with “You have the right to 
[[access and]] delete your data”, so “You have the right to” and “your data” is included in both annotations). 
Structure your annotations as JSON objects in a list. Each object should have the following keys: - "requirement": 
The transparency requirement that the annotated word or phrase fulfils. - "value": The annotated word or phrase. - 
"performed": A boolean indicating whether the stated activity is performed or explicitly not performed (e.g. "we do 
not collect your name").'''.replace('\n', '')

example_1_in = '''{ "type": "list_item", "context": [ { "text": "Amazon.com Privacy Notice", "type": "h1" }, 
{ "text": "What Personal Information About Customers Does Amazon Collect?", "type": "h2" }, { "text": "Here are the 
types of personal information we collect:", "type": "list_intro" } ], "passage": "Information from Other Sources: We 
might receive information about you from other sources, such as updated delivery and address information from our 
carriers, which we use to correct our records and deliver your next purchase more easily. We do not, however, 
collect your gender and sexual orientation. Click here to see additional examples of the information we receive. This 
information is stored on servers in the USA. You may request that we delete your personal information at any time." 
}'''.replace('\n', '')

example_1_out = '''[ { "requirement": "Data Categories", "value": "updated delivery [[and address]] information", "performed": 
true }, { "requirement": "Data Categories", "value": "updated [[delivery and]] address information", "performed": true }, 
{ "requirement": "Source of Data", "value": "our carriers", "performed": true }, { "requirement": "Processing 
Purpose", "value": "correct our records", "performed": true }, { "requirement": "Processing Purpose", 
"value": "deliver your next purchase more easily", "performed": true }, { "requirement": "Data Categories", 
"value": "sexual orientation", "performed": false }, { "requirement": "Third-country Transfers", "value": "USA", 
"performed": true }, { "requirement": "Right to Erasure", "value": "You may request that we delete your personal 
information at any time", "performed": true } ]'''.replace('\n', '')

example_2_in = '''{ "type": "list_item", "context": [ { "text": "Amazon.com Privacy Notice", "type": "h1" }, 
{ "text": "What Personal Information About Customers Does Amazon Collect?", "type": "h2" }, { "text": "Here are the 
types of personal information we collect:", "type": "list_intro" } ], "passage": "Information from Other Sources: We 
might receive information about you from other sources, such as updated delivery and address information from our 
carriers, which we use to correct our records and deliver your next purchase more easily. We do not, however, 
collect your gender and sexual orientation. Click here to see additional examples of the information we receive. 
Please not that this information is stored on servers in the USA. You may request that we delete your personal 
information at any time." }'''.replace('\n', '')

example_2_out = '''[ { "requirement": "Controller Name", "value": "Amazon", "performed": true }, 
{ "requirement": "Data Categories", "value": "delivery and address information", "performed": true }, 
{ "requirement": "Source of Data", "value": "our carriers", "performed": true }, { "requirement": 
"Processing Purpose", "value": "correct our records", "performed": true }, { "requirement": "Processing 
Purpose", "value": "deliver your next purchase more easily", "performed": true }, { "requirement": "Data 
Categories", "value": "gender", "performed": false }, { "requirement": "Data Categories", "value": 
"sexual orientation", "performed": false }, { "requirement": "Data Categories", "value": "gender", 
"performed": false }, { "requirement": "Right to Erasure", "value": "You may request that we delete your 
personal information at any time", "performed": true } ]"'''.replace('\n', '')

example_3_in = ''''''

example_3_out = ''''''

example_4_in = ''''''

example_4_out = ''''''

task = "annotator"
model = api_wrapper.models['gpt-4o-mini']


def execute(run_id: str, pkg: str, in_folder: str, out_folder: str, use_batch: bool = False):
    print(f"Annotating {pkg}...")

    policy = util.load_policy_json(run_id, pkg, 'json')
    if policy is None:
        return None

    output = []
    total_cost = 0

    for index, passage in enumerate(policy):
        if use_batch:
            result, cost = api_wrapper.retrieve_batch_result_entry(run_id, task, f"{run_id}_{task}_{pkg}_{index}")
        else:
            result, cost = api_wrapper.prompt(
                run_id=run_id,
                pkg=pkg,
                task=task,
                model=model,
                system_msg=system_msg,
                user_msg=json.dumps(passage),
                examples=[(example_1_in, example_1_out)]
            )

        total_cost += cost
        try:
            passage['annotations'] = json.loads(result)
        except json.JSONDecodeError:
            print(result)
            raise json.JSONDecodeError
        output.append(passage)

    print(f"Annotation cost: {total_cost}")

    with open(f"output/{run_id}/gpt_responses/costs_annotator.csv", "a") as f:
        f.write(f"{pkg},{total_cost}\n")

    util.write_to_file(f"output/{run_id}/annotated/{pkg}.json", json.dumps(output, indent=4))


def prepare_batch(run_id: str, pkg: str, in_folder: str):
    policy = util.load_policy_json(run_id, pkg, 'json')
    if policy is None:
        return None

    entries = []

    for index, passage in enumerate(policy):
        entry = api_wrapper.prepare_batch_entry(
            run_id=run_id,
            pkg=pkg,
            task=task,
            model=model,
            system_msg=system_msg,
            user_msg=json.dumps(passage),
            examples=[(example_1_in, example_1_out)],
            max_tokens=8192,
            entry_id=index
        )
        entries.append(entry)

    return entries
