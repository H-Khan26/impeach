Here’s a clean markdown view of all the embedded contradictions so you can directly compare against your pipeline’s output.

---

## Document Labels

- **Doc 1:** *Mason v. Billy* (`data/input/Mason_billy.jsonl`, `doc_id: 001`)
- **Doc 2:** *Mason v. Watson* (`data/input/Mason_watson.jsonl`, `doc_id: 002`)

---

## 1. Interdocument Contradictions (Doc 1 vs Doc 2)

| ID   | Type(s) (Structural)                                                                 | Doc 1 – Mason v. Billy                                                                 | Doc 2 – Mason v. Watson                                                               | Notes / What Contradicts |
| ---- | ------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | ------------------------ |
| C-01 | Location (value_type: `LOCATION`, property_type: `LOCATION`)                          | **Eastvale, California** (`LOC-001`)                                                   | **Stockton, California** (`LOC-002`)                                                  | Same event framing (“attending a fair”), different city. |
| C-02 | Attribute / Categorical (value_type: `CATEGORICAL`, property_type: `ATTRIBUTE`)       | Played **beer pong** (`OTHER-003`)                                                     | Played **cup pong** (`OTHER-004`)                                                     | Activity differs. |
| C-03 | Relation / Entity Ref (value_type: `ENTITY_REF`, property_type: `RELATION`)           | Struck by **Billy** (`PER-002`)                                                        | Struck by **Watson** (`PER-003`)                                                      | Different alleged assailant. |
| C-04 | Attribute / Categorical (value_type: `CATEGORICAL`, property_type: `ATTRIBUTE`)       | **Severe** concussion                                                                  | **Minor** concussion                                                                  | Severity differs. |
| C-05 | Temporal / Duration (value_type: `TEMPORAL_INTERVAL`, property_type: `TEMPORAL`)      | Expenses spanned **several months**                                                    | Expenses spanned **2 months**                                                         | Duration differs. |
| C-06 | Quantitative / Money (value_type: `NUMERIC`, property_type: `QUANTITATIVE`)           | Total hospital expenses **$3200** (`MONEY-001`)                                         | Total hospital expenses **$300** (`MONEY-002`)                                         | Amount differs. |

---

## 2. Atomic Facts (with structural types)

All facts below use **domain-agnostic structural types** from `contradiction_engine/config/mappings/fact_categorization.yaml`:
- `temporal_context`: `current_incident` (this scenario’s incident)
- `value_type`: one of `NUMERIC, TEMPORAL_POINT, TEMPORAL_INTERVAL, LOCATION, BOOLEAN, CATEGORICAL, ENTITY_REF, TEXT`
- `property_type`: one of `TEMPORAL, QUANTITATIVE, LOCATION, ATTRIBUTE, RELATION, EXISTENCE, MODALITY`

### Doc 1 – Mason v. Billy (`data/input/Mason_billy.jsonl`)

**Source text (verbatim)**:
> Mason was attending a fair in Eastvale, California on July 8, 2022. He was playing beer pong when he was struck in the head by a large clown named Billy. Mason suffered a severe concussion and brain damage. His hospital expenses from the injury spanned several months and summed up to $3200.

| Fact ID   | Subject (Entity)      | Predicate (normalized)        | Object / Value                         | temporal_context     | value_type          | property_type   | Linked entities |
| --------- | --------------------- | ----------------------------- | -------------------------------------- | -------------------- | ------------------ | -------------- | -------------- |
| D1-F-001  | Mason (`PER-001`)     | attended_fair_in              | Eastvale, California (`LOC-001`)       | current_incident     | LOCATION           | LOCATION       | `PER-001, LOC-001` |
| D1-F-002  | Mason (`PER-001`)     | attended_fair_on              | July 8, 2022 (`DATE-001`)              | current_incident     | TEMPORAL_POINT     | TEMPORAL       | `PER-001, DATE-001` |
| D1-F-003  | Mason (`PER-001`)     | was_playing                   | beer pong (`OTHER-003`)                | current_incident     | CATEGORICAL        | ATTRIBUTE      | `PER-001, OTHER-003` |
| D1-F-004  | Mason (`PER-001`)     | was_struck_in_head_by         | Billy (`PER-002`)                      | current_incident     | ENTITY_REF         | RELATION       | `PER-001, PER-002` |
| D1-F-005  | Mason (`PER-001`)     | concussion_severity            | severe                                 | current_incident     | CATEGORICAL        | ATTRIBUTE      | `PER-001` |
| D1-F-006  | Mason (`PER-001`)     | brain_damage_occurred          | true                                   | current_incident     | BOOLEAN            | EXISTENCE      | `PER-001` |
| D1-F-007  | Mason (`PER-001`)     | hospital_expenses_spanned      | several months                         | current_incident     | TEMPORAL_INTERVAL  | TEMPORAL       | `PER-001` |
| D1-F-008  | Mason (`PER-001`)     | hospital_expenses_total        | $3200 (`MONEY-001`)                    | current_incident     | NUMERIC            | QUANTITATIVE   | `PER-001, MONEY-001` |

### Doc 2 – Mason v. Watson (`data/input/Mason_watson.jsonl`)

**Source text (verbatim)**:
> Mason was attending a fair in Stockton, California on July 8, 2022. He was playing cup pong when he was struck in the head by a large clown named Watson. Mason suffered a minor concussion. His hospital expenses from the injury spanned 2 months and summed up to $300

| Fact ID   | Subject (Entity)      | Predicate (normalized)        | Object / Value                         | temporal_context     | value_type          | property_type   | Linked entities |
| --------- | --------------------- | ----------------------------- | -------------------------------------- | -------------------- | ------------------ | -------------- | -------------- |
| D2-F-001  | Mason (`PER-001`)     | attended_fair_in              | Stockton, California (`LOC-002`)       | current_incident     | LOCATION           | LOCATION       | `PER-001, LOC-002` |
| D2-F-002  | Mason (`PER-001`)     | attended_fair_on              | July 8, 2022 (`DATE-001`)              | current_incident     | TEMPORAL_POINT     | TEMPORAL       | `PER-001, DATE-001` |
| D2-F-003  | Mason (`PER-001`)     | was_playing                   | cup pong (`OTHER-004`)                 | current_incident     | CATEGORICAL        | ATTRIBUTE      | `PER-001, OTHER-004` |
| D2-F-004  | Mason (`PER-001`)     | was_struck_in_head_by         | Watson (`PER-003`)                     | current_incident     | ENTITY_REF         | RELATION       | `PER-001, PER-003` |
| D2-F-005  | Mason (`PER-001`)     | concussion_severity            | minor                                  | current_incident     | CATEGORICAL        | ATTRIBUTE      | `PER-001` |
| D2-F-006  | Mason (`PER-001`)     | hospital_expenses_spanned      | 2 months                               | current_incident     | TEMPORAL_INTERVAL  | TEMPORAL       | `PER-001` |
| D2-F-007  | Mason (`PER-001`)     | hospital_expenses_total        | $300 (`MONEY-002`)                     | current_incident     | NUMERIC            | QUANTITATIVE   | `PER-001, MONEY-002` |

---

## 3. Entity Key (canonical `entity_types`)

Entity types come from `contradiction_engine/config/mappings/entity_types.yaml`:
`PER, ORG, LOC, CASE, STATUTE, COURT, DOCKET_ENTRY, EXHIBIT, EXHIBIT_ID, DATE, TIME, DOC_LOCAL, PRODUCT, FACILITY, MONEY, OTHER`

### PER – Person

| Entity ID | Type | Canonical Name | Mentions / Aliases (non-exhaustive) |
| --------- | ---- | -------------- | ----------------------------------- |
| PER-001   | PER  | Mason          | “Mason”; “He” (when referring to Mason) |
| PER-002   | PER  | Billy          | “Billy”; “a large clown named Billy”; “clown named Billy” |
| PER-003   | PER  | Watson         | “Watson”; “a large clown named Watson”; “clown named Watson” |

### LOC – Location

| Entity ID | Type | Canonical Name        | Mentions / Aliases (non-exhaustive) |
| --------- | ---- | --------------------- | ----------------------------------- |
| LOC-001   | LOC  | Eastvale, California  | “Eastvale, California”; “Eastvale” |
| LOC-002   | LOC  | Stockton, California  | “Stockton, California”; “Stockton” |
| LOC-003   | LOC  | California            | “California” |

### DATE – Date References

| Entity ID | Type | Canonical Date | Mentions / Aliases (non-exhaustive) |
| --------- | ---- | -------------- | ----------------------------------- |
| DATE-001  | DATE | 2022-07-08     | “July 8, 2022” |

### MONEY – Monetary Amounts

| Entity ID | Type  | Canonical Amount | Mentions / Aliases (non-exhaustive) |
| --------- | ----- | ---------------- | ----------------------------------- |
| MONEY-001 | MONEY | 3200 USD         | “$3200”; “3200”; “$ 3200” |
| MONEY-002 | MONEY | 300 USD          | “$300”; “300”; “$ 300” |

### DOC_LOCAL – Local Documents (test inputs)

| Entity ID | Type      | Canonical Name | Mentions / Aliases (non-exhaustive) |
| --------- | --------- | -------------- | ----------------------------------- |
| DOC-001   | DOC_LOCAL | Mason_billy    | `data/input/Mason_billy.jsonl`; “Doc 1” |
| DOC-002   | DOC_LOCAL | Mason_watson   | `data/input/Mason_watson.jsonl`; “Doc 2” |

### OTHER – Everything Else

| Entity ID | Type  | Canonical Name      | Mentions / Aliases (non-exhaustive) |
| --------- | ----- | ------------------- | ----------------------------------- |
| OTHER-001 | OTHER | The fair (Doc 1)    | “a fair” (in Eastvale context) |
| OTHER-002 | OTHER | The fair (Doc 2)    | “a fair” (in Stockton context) |
| OTHER-003 | OTHER | Beer pong           | “beer pong” |
| OTHER-004 | OTHER | Cup pong            | “cup pong” |