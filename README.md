<h1 align = "center">LEND</h1>
<h2 align = "center">Loan Eligibility & Navigation Decisioning </h2>
<p align = "justify">LEND accelerates loan approvals with precision by leveraging Lorentzian Neural Networks to optimize risk assessment and decision making.

<p align="center">
  <b>General Approval Guidelines (TL;DR)</b>
</p>

<table align="center">
  <tr>
    <th>Factor</th>
    <th>Ideal Range for Approval</th>
    <th>High-Risk / Rejection Threshold</th>
  </tr>
  <tr>
    <td align="center"><b>Age</b></td>
    <td align="center">21 – 65 years</td>
    <td align="center">Below 18 or above 65 (without proof of income)</td>
  </tr>
  <tr>
    <td align="center"><b>Annual Income ($)</b></td>
    <td align="center">At least 3-5× EMI</td>
    <td align="center">Low income with high DTI (&gt;50%)</td>
  </tr>
  <tr>
    <td align="center"><b>Education Level</b></td>
    <td align="center">Bachelor's or higher</td>
    <td align="center">High school or lower (for large loans)</td>
  </tr>
  <tr>
    <td align="center"><b>Home Ownership</b></td>
    <td align="center">Owns or has a mortgage</td>
    <td align="center">Renting (unless strong financials)</td>
  </tr>
  <tr>
    <td align="center"><b>DTI Ratio (%)</b></td>
    <td align="center">Below 40% (max 50%)</td>
    <td align="center">Above 50% (almost always rejected)</td>
  </tr>
  <tr>
    <td align="center"><b>Interest Rate (%)</b></td>
    <td align="center">6% - 15%</td>
    <td align="center">Above 25% = high risk</td>
  </tr>
  <tr>
    <td align="center"><b>Loan Intent</b></td>
    <td align="center">Home, education, debt consolidation</td>
    <td align="center">Personal (large amounts), business loans (without collateral)</td>
  </tr>
  <tr>
    <td align="center"><b>Previous Defaults</b></td>
    <td align="center">None or 1 minor (paid off)</td>
    <td align="center">1+ unpaid default = almost always rejected</td>
  </tr>
</table>


- Loan approval takes around 3-4 days from acceptance to approval, we are aiming to provide that in just a couple of hours (and will work it through for making it to minutes), without tampering with the accuracy.
- Our model provides a hybrid structure integrating Lorentzian Neural Networks along with Ensemble Methods for accuracy boost to ensure that the loan approval is handled with the atmost care.

<p align="center">
  <b>Credit Score Categories & Loan Approval Chances</b>
</p>

<table align="center">
  <tr>
    <th>Credit Score</th>
    <th>Category</th>
    <th>Loan Approval Chances</th>
  </tr>
  <tr>
    <td align="center"><b>800 - 850</b></td>
    <td align="center">Exceptional</td>
    <td align="center">Almost guaranteed approval with best interest rates</td>
  </tr>
  <tr>
    <td align="center"><b>740 - 799</b></td>
    <td align="center">Very Good</td>
    <td align="center">Very high approval rate, low-interest rates</td>
  </tr>
  <tr>
    <td align="center"><b>670 - 739</b></td>
    <td align="center">Good</td>
    <td align="center">Standard approval, decent rates</td>
  </tr>
  <tr>
    <td align="center"><b>580 - 669</b></td>
    <td align="center">Fair</td>
    <td align="center">High interest rates, may need a co-signer/collateral</td>
  </tr>
  <tr>
    <td align="center"><b>Below 580</b></td>
    <td align="center">Poor</td>
    <td align="center">Very low chance of approval without collateral</td>
  </tr>
</table>

</p>

<img src = "README/LorentzNN.png">


<img src = "README/ensemble-model-diagram.png">

<p align = "justify">Model architecture and overall flow for the model.</p>




References

```
https://papertalk.org/papertalks/6134
```

```
https://www.researchgate.net/figure/Fitted-Lorentzian-curve-fitting-LCF-method-on-a-noisy-Brillouin-gain-spectrum-BGS_fig3_344655338
```
