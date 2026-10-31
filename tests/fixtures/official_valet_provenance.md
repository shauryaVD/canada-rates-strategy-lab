# Official Valet fixture provenance

`official_valet_observations.json` is an unmodified response fetched from the Bank
of Canada Valet API on October 31, 2026. It is a recorded real-data fixture for
parser tests, never used as a substitute for live application data.

Request: <https://www.bankofcanada.ca/valet/observations/BD.CDN.2YR.DQ.YLD,BD.CDN.3YR.DQ.YLD,BD.CDN.5YR.DQ.YLD,BD.CDN.7YR.DQ.YLD,BD.CDN.10YR.DQ.YLD/json?start_date=2026-10-26&end_date=2026-10-30>

Source documentation: <https://www.bankofcanada.ca/valet-api-how-to/>

Units: percent. Frequency: business daily. These are selected Government of Canada
benchmark bond yields, not zero-coupon yields. The response contains observations
through October 29, 2026; request end date does not imply publication through that
date. Parser edge-case tests explicitly mutate copies of this recorded fixture.
