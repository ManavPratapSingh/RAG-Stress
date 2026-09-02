"""Passage rendering engine (SPEC §4.4).

Renders facts into natural language passages using attribute paraphrase templates
padded with neutral filler sentences to satisfy length 60-180 words.
"""

import random
from datetime import date

from ragstress.kb.facts import Fact, Passage

TEMPLATES: dict[str, list[str]] = {
    "notice_period": [
        "Under standard operational guidelines, the notice period for {entity_name} is established at {value}.",
        "Supervisors and team members must provide a notice period of {value} in accordance with {entity_name} requirements.",
        "The required notice period specified by {entity_name} is exactly {value} before any formal transition takes effect.",
        "Per administrative provisions, {entity_name} mandates an official notice period of {value} for all recorded filings.",
    ],
    "stipend_amount": [
        "The allocated monthly stipend under {entity_name} is set at ${value}.",
        "Eligible participants receive a monthly stipend of ${value} according to {entity_name} terms.",
        "Documentation for {entity_name} indicates an approved monthly stipend amounting to ${value}.",
        "Financial governance schedules for {entity_name} define the regular monthly stipend as ${value}.",
    ],
    "approval_threshold": [
        "Any expenditure exceeding the approval threshold of ${value} requires formal authorization under {entity_name}.",
        "For {entity_name}, the standard approval threshold is pegged at ${value} for department disbursements.",
        "Under the governance rules of {entity_name}, the official approval threshold is ${value}.",
        "Transactions subject to {entity_name} maintain an established approval threshold of ${value}.",
    ],
    "review_cycle": [
        "The governance committee conducts a regular review cycle of {value} for {entity_name}.",
        "In compliance with institutional audit standards, the review cycle for {entity_name} is fixed at {value}.",
        "Administrative oversight mandates a review cycle of {value} across all operations governed by {entity_name}.",
        "The operational schedule for {entity_name} designates an official review cycle of {value}.",
    ],
    "max_throughput": [
        "Benchmarked laboratory testing confirms the maximum throughput of {entity_name} reaches {value} requests/s.",
        "Engineered for high-volume environments, {entity_name} delivers a maximum throughput of {value} requests/s.",
        "System telemetry specifies that the maximum throughput for {entity_name} is rated at {value} requests/s.",
        "Under peak operational load, {entity_name} achieves a verified maximum throughput of {value} requests/s.",
    ],
    "warranty_period": [
        "Standard enterprise support agreements grant a warranty period of {value} on {entity_name}.",
        "Hardware reliability documentation certifies the warranty period for {entity_name} as {value}.",
        "Every deployed unit of {entity_name} includes a comprehensive warranty period of {value}.",
        "The manufacturer warranty period applicable to {entity_name} is explicitly set to {value}.",
    ],
    "list_price": [
        "Official catalog listings indicate the suggested list price for {entity_name} is ${value}.",
        "The commercial list price published for {entity_name} is established at ${value}.",
        "Enterprise procurement schedules quote a standard list price of ${value} for {entity_name}.",
        "Unit sales for {entity_name} begin with an authorized list price of ${value}.",
    ],
    "max_operating_temp": [
        "Thermal tolerance tests verify that the maximum operating temperature of {entity_name} is {value} °C.",
        "Environmental enclosures must ensure the maximum operating temperature for {entity_name} does not exceed {value} °C.",
        "Specifications cite a maximum operating temperature of {value} °C under sustained operational stress for {entity_name}.",
        "According to engineering datasheets, {entity_name} has a rated maximum operating temperature of {value} °C.",
    ],
    "capacity": [
        "Architectural documentation confirms an authorized capacity of {value} people within {entity_name}.",
        "Occupancy safety permits certify that {entity_name} operates with a maximum capacity of {value} people.",
        "The primary infrastructure of {entity_name} supports an overall capacity of {value} people.",
        "Facility management records verify an established capacity of {value} people for {entity_name}.",
    ],
    "opening_date": [
        "Official institutional records document the formal opening date for {entity_name} as {value}.",
        "Commissioning was completed and the opening date of {entity_name} was formally recorded as {value}.",
        "Operations commenced following the scheduled opening date of {entity_name} on {value}.",
        "Corporate archives establish the opening date for {entity_name} as {value}.",
    ],
    "floor_count": [
        "Structural schematics indicate the number of floors within {entity_name} is {value}.",
        "The physical building of {entity_name} encompasses an official number of floors equal to {value}.",
        "Building permits verify that {entity_name} features an approved number of floors of {value}.",
        "Engineering records list the number of floors for {entity_name} as {value}.",
    ],
    "security_level": [
        "Access control protocols classify {entity_name} under an assigned security level of {value}.",
        "Site security assessments confirm that the security level for {entity_name} is currently designated as {value}.",
        "Compliance guidelines establish that {entity_name} maintains a mandatory security level of {value}.",
        "Facility monitoring regulations define the operational security level of {entity_name} as {value}.",
    ],
}

NEUTRAL_FILLERS: list[str] = [
    "All associated operational procedures remain subject to regular supervisory oversight and scheduled administrative reviews.",
    "Departmental liaisons ensure cross-functional alignment across all participating teams throughout the fiscal period.",
    "Routine maintenance protocols are logged in central archives to verify ongoing procedural compliance.",
    "Authorized personnel must retain documentation of routine activities in accordance with standardized corporate governance rules.",
    "Internal auditing teams conduct periodic spot-checks to guarantee that standards align with corporate policy directives.",
    "Stakeholders receive quarterly briefing packets summarizing general procedural updates and organizational metrics.",
    "Standard operating manuals are distributed to all working groups to maintain procedural uniformity.",
    "Quality assurance frameworks provide structured feedback mechanisms for all participating divisions.",
    "Operational telemetry is monitored continuously to guarantee adherence to institutional guidelines.",
    "Collaborating teams meet on a scheduled cadence to review operational performance against baseline expectations.",
    "Supervisory staff members complete mandatory compliance attestations prior to the deployment of new operational directives.",
    "Central coordinators maintain complete administrative logs regarding all related departmental infrastructure assets.",
]


def render_fact_sentence(fact: Fact, rng: random.Random) -> str:
    """Render a single fact into a sentence using one of the paraphrase templates."""
    templates = TEMPLATES.get(fact.attribute)
    if not templates:
        raise ValueError(f"No templates defined for attribute '{fact.attribute}'")
    template = rng.choice(templates)
    return template.format(entity_name=fact.entity_name, value=fact.value)


def render_passage(
    doc_id: str,
    title: str,
    facts: list[Fact],
    status: str,
    effective_date: date,
    rng: random.Random,
    min_words: int = 65,
    max_words: int = 175,
) -> Passage:
    """Render 1-3 facts into a passage of length between 60-180 words.

    Padded with neutral filler sentences.
    One single renderer used for all passages (gold, twins, distractors) to prevent style tells.
    """
    if not facts:
        raise ValueError("Passage must contain at least 1 fact")
    if len(facts) > 3:
        raise ValueError("Passage must contain at most 3 facts")

    # Render fact sentences
    fact_sentences = [render_fact_sentence(f, rng) for f in facts]

    # Deterministic filler selection
    shuffled_fillers = list(NEUTRAL_FILLERS)
    rng.shuffle(shuffled_fillers)

    sentences = list(fact_sentences)
    current_text = " ".join(sentences)
    word_count = len(current_text.split())

    filler_idx = 0
    while word_count < min_words and filler_idx < len(shuffled_fillers):
        sentences.append(shuffled_fillers[filler_idx])
        filler_idx += 1
        current_text = " ".join(sentences)
        word_count = len(current_text.split())

    # Ensure length does not exceed max_words
    # If a filler pushed it over max_words, we can drop the last filler if still >= 60 words
    while word_count > max_words and len(sentences) > len(fact_sentences):
        sentences.pop()
        current_text = " ".join(sentences)
        word_count = len(current_text.split())

    return Passage(
        doc_id=doc_id,
        title=title,
        text=current_text,
        status=status,  # type: ignore[arg-type]
        effective_date=effective_date,
        facts=facts,
    )
