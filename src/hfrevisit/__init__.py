"""hfrevisit — reproducible re-analysis of heart-failure survival prediction.

Re-evaluates Chicco & Jurman (2020, BMC Med Inform Decis Mak 20:16) on the
Ahmad et al. (2017) heart-failure cohort, showing that the follow-up ``time``
variable leaks the outcome, re-framing the task as time-to-event analysis,
and externally validating the resulting models on MIMIC-III.
"""

__version__ = "1.0.0"
