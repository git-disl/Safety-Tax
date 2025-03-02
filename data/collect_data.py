from collections import Counter
from functools import partial
import random

import datasets

from decontaminate_util import *

BAD_OMNIMATH_SAMPLES = [
    {"question": "Let $\\mathbb{R}$ be the set of real numbers .  Determine all functions $f\u00a0: \\mathbb{R} \\rightarrow \\mathbb{R}$ such that\n  \nfor all pairs of real numbers $x$ and $y$ ."},
    {"question": "Find the sum of the ages of everyone who wrote a problem for this year's HMMT November contest. If your answer is $X$ and the actual value is $Y$, your score will be $\\max (0,20-|X-Y|)$"},
]

TMP_QUESTIONS_TO_KEEP = [
    "During the Northern and Southern Dynasties, there was a mathematical problem in the ancient book \"Zhang Qiujian's Arithmetic\": \"There are ten ranks of people today. For each rank, the palace grants gold with a decreasing arithmetic sequence. The top three people received a total of 4 pounds of gold; the last four people each received 3 pounds of gold; and the remaining three people in the middle ranks will also receive gold according to the arithmetic sequence. Question: How many more pounds of gold does each rank receive than the rank below it?\"\n\nA: $\\frac{4}{39}$\nB: $\\frac{7}{78}$\nC: $\\frac{7}{76}$\nD: $\\frac{5}{85}$",
    "Given the rest of reaction components:\nreactant: Cc1ccc2c(cnn2C2CCCCO2)c1B1OC(C)(C)C(C)(C)O1\nligand: CC1(C)c2cccc(P(c3ccccc3)c3ccccc3)c2Oc2c(P(c3ccccc3)c3ccccc3)cccc21\nsolvent: CO\nbase: [Li+].CC(C)(C)[O-]  \nReactants list for selection:\nIc1ccc2ncccc2c1,Brc1ccc2ncccc2c1,Clc1ccc2ncccc2c1\nWhat is the optimal reactant?",
    "Laparoscopic sleeve gastrectomy (LSG) is currently being performed with increasing frequency worldwide. It offers an excellent weight loss and resolution of comorbidities in the short term with a very low incidence of complications. However, the ever present risk of a staple line leak is still a major concern.\nSince 2005, data from obese patients that undergo bariatric procedures in Germany are prospectively registered in an online database and analyzed at the Institute of Quality Assurance in Surgical Medicine. For the current analysis, all patients that had undergone primary sleeve gastrectomy for morbid obesity within a 7-year period were considered.\nUsing the GBSR, data from 5.400 LSGs were considered for analysis. Staple line leak rate decreased during the study period from 6.5 to 1.4 %. Male gender, higher BMI, concomitant sleep apnea, conversion to laparotomy, longer operation time, use of both buttresses and oversewing, and the occurrence of intraoperative complications were associated with a significantly higher leakage rate. On multivariate analysis, operation time and year of procedure only had a significant impact on staple line leak rate.\n\nAre there risk factors that increase the rate of staple line leakage in patients undergoing primary sleeve gastrectomy for morbid obesity?",
    "Person A and person B each guess a riddle in each guessing activity. If one person guesses correctly and the other person guesses incorrectly, the person who guessed correctly wins; otherwise, it is a tie. It is known that in each activity, the probabilities of A and B guessing correctly are $\\frac{5}{6}$ and $\\frac{3}{5}$, respectively. The outcomes of A and B guessing correctly or incorrectly do not affect each other within each activity, and each activity is independent of others. The probability of A winning in one activity is ______; the probability of A winning at least 2 out of 3 activities is ______.",
    "How many nonmetals are gases?\n\nA. 9 nonmetal gases\nB. 11 nonmetal gases\nC. 7 nonmetal gases\nD. 13 nonmetal gases",
    "Updated guidelines for the screening and management of cervical cancer in the United States recommend starting Papanicolaou (Pap) testing at age 21 and screening less frequently with less aggressive management for abnormalities. We sought to examine updated Pap test screening guidelines and how they may affect the detection of invasive cervical cancer, especially among women<30 years of age.\nPatients diagnosed at Brigham and Women's Hospital with invasive cervical cancer between 2002 and 2012 were retrospectively identified. Prior screening history was obtained and patients were divided into two groups based on age<30 years or age \u226530 years. The two groups were then compared with respect to demographics, pathological findings, and time to diagnosis.\nA total of 288 patients with invasive cervical carcinoma were identified. Among these patients, 109 had adequate information on prior screening history. Invasive adenocarcinoma (IAC) was diagnosed in 37 (33.94%) patients, whereas 64 (58.72%) patients were diagnosed with invasive squamous cell carcinoma (ISCC). The remaining eight patients were diagnosed with other types of cancers of the cervix. A total of 13 patients were younger than 30 while 96 patients were 30 or older. The mean time from normal Pap to diagnosis of IAC was 15 months in patients younger than 30 years of age compared to 56 months in patients aged 30 and older (p\u2009<\u20090.001). The mean time from normal Pap to diagnosis of ISCC was 38 months in patients younger than 30 years of age and 82 months in patients aged 30 and older (p\u2009=\u20090.018).\n\nScreening History Among Women with Invasive Cervical Cancer in an Academic Medical Center: Will We Miss Cancers Following Updated Guidelines?",
    "Suppose a seamount forms from an underwater volcano. Birds on the mainland colonized the island. How might this lead to speciation?\n\nA. Birds adapt to the island's conditions through random mutations, leading to new species with characteristics suited to their environment.\nB. Birds bring mainland plants, altering the island's ecosystem and causing new species to emerge.\nC. Birds interbreed with local species, creating hybrids that eventually form a new species.\nD. Birds introduce diseases, causing the extinction of some species and the emergence of new ones.",
    "What does the second law of thermodynamics say about entropy?\n\nA. Entropy increases in natural processes.\nB. Entropy is unrelated to natural processes.\nC. Entropy decreases in natural processes.\nD. Entropy remains constant in natural processes.",
    "Given four lines such that any three of them determine a triangle, prove that the orthocenters of these four triangles are collinear.",
    "The graph of the function $y=\\cos x - \\sin x$ has a line of symmetry given by (__).\n\nA: $x= \\frac{\\pi}{4}$\nB: $x= \\frac{\\pi}{8}$\nC: $x= -\\frac{\\pi}{8}$\nD: $x= -\\frac{\\pi}{4}$",
    "To investigate whether the Patient Health Questionnaire-9 (PHQ-9) possesses the essential psychometric characteristics to measure depressive symptoms in people with visual impairment.\nThe PHQ-9 scale was completed by 103 participants with low vision. These data were then assessed for fit to the Rasch model.\nThe participants' mean +/- standard deviation (SD) age was 74.7 +/- 12.2 years. Almost one half of them (n = 46; 44.7%) were considered to have severe vision impairment (presenting visual acuity<6/60 in the better eye). Disordered thresholds were evident initially. Collapsing the two middle categories produced ordered thresholds and fit to the Rasch model (chi = 10.1; degrees of freedom = 9; p = 0.34). The mean (SD) items and persons Fit Residual values were -0.31 (1.12) and -0.25 (0.78), respectively, where optimal fit of data to the Rasch model would have a mean = 0 and SD = 1. Unidimensionality was demonstrated confirming the construct validity of the PHQ-9 and there was no evidence of differential item functioning on a number of factors including visual disability. The person separation reliability value was 0.80 indicating that the PHQ-9 has satisfactory precision. There was a degree of mistargeting as expected in this largely non-clinically depressed sample.\n\nCan clinicians use the PHQ-9 to assess depression in people with vision loss?",
    "Given the rest of reaction components:\nreactant: Cc1ccc2c(cnn2C2CCCCO2)c1[B-](F)(F)F.[K+]\nligand: C1=C[C@H]2[Fe][C@@H]1[C]2P(c1ccccc1)c1ccccc1\nsolvent: C1CCOC1\nbase: [F-].[Cs+]  \nReactants list for selection:\nIc1ccc2ncccc2c1,Brc1ccc2ncccc2c1,Clc1ccc2ncccc2c1\nWhat is the optimal reactant?",
    "A 45-year-old homeless man comes to the emergency department because of progressive neck pain for 3 days. He also reports headaches and numbness in both hands. Over the past 4 months, he has had intermittent lower back pain after waking up. The back pain improves with movement. He also has recurrent episodes of gout in the metatarsophalangeal joint of his right big toe. He has smoked one pack of cigarettes daily for 20 years and drinks four beers daily. The patient is a known user of intravenous heroin. He appears acutely ill. His temperature is 39\u00b0C (102.2\u00b0F), pulse is 110/min, and blood pressure is 140/85 mm Hg. There are several track marks on both forearms. Examination of the neck shows warmth, erythema, and limited range of motion. Gentle palpation over the midcervical spinal processes causes severe pain. Laboratory studies show:\nHemoglobin 11 g/dL\nLeukocyte count 14,200/mm3\nSegmented neutrophils 77%\nEosinophils 1%\nLymphocytes 20%\nMonocytes 2%\nPlatelet count 278,000/mm3\nErythrocyte sedimentation rate 54 mm/h\nBlood cultures are pending. An x-ray of the cervical spine shows no abnormalities. An MRI of the spine shows signs of inflammation. A bone biopsy confirms the diagnosis. Which of the following is the most appropriate next step in management?\"\n\nA. Lumbar puncture\nB. Intravenous ciprofloxacin and vancomycin therapy\nC. Oral indomethacin therapy\nD. Bone scintigraphy\n\"",
    "To study whether nontriploid partial hydatidiform moles truly exist.\nWe conducted a reevaluation of pathology and ploidy in 19 putative nontriploid partial hydatidiform moles using standardized histologic diagnostic criteria and repeat flow cytometric testing by the Hedley technique.\nOn review of the 19 moles, 53% (10/19) were diploid nonpartial moles (initially pathologically misclassified), and 37% (7/19) were triploid partial moles (initial ploidy misclassifications). One additional case (5%) was a diploid early complete mole (initially pathologically misclassified).\n\nDo nontriploid partial hydatidiform moles exist?",
    "A curious tourist wants to walk through the streets of the Old Town from the train station (point $A$ on the map) to their hotel (point $B$). The tourist wants their route to be as long as possible, but they are not interested in visiting the same intersection twice, so they do not do so. Draw the longest possible route on the map and prove that there is no longer route.",
    "A 35-year-old man comes to the physician because of episodes of difficulty swallowing for the past 3 months. He feels solid food getting stuck in his chest behind the sternum when he eats. Drinking does not cause any difficulty swallowing. He has no coughing or nasal regurgitation. He has no hoarseness or weight loss. He has had heartburn for 2 years with no response to high-dose omeprazole. His past medical history is also significant for asthma and eczema. He takes no medications except for omeprazole. His vital signs are within normal limits. Physical examination shows no abnormal findings. Which of the following best explains these findings?\n\nA. Achalasia\nB. Diffuse esophageal spasm\nC. Eosinophilic esophagitis\nD. Esophageal reflux disease",
    "Given that the three sides of triangle $\\triangle ABC$ are $a$, $b$, and $c$, which of the following conditions cannot determine that $\\triangle ABC$ is a right triangle?\n\nA: $\\angle A + \\angle B = \\angle C$\n\nB: $a^{2} = 5$, $b^{2} = 12$, $c^{2} = 13$\n\nC: $\\angle A : \\angle B : \\angle C = 1 : 1 : 2$\n\nD: $a = 7$, $b = 24$, $c = 25$",
    "If the degree of the central angle corresponding to an arc increases by $1^{\\circ}$, and the radius of the arc is $R$, then the arc length increases by ( )\n\nA: $\\frac{\u03c0R}{360}$\n\nB: $\\frac{180}{\u03c0R}$\n\nC: $\\frac{360}{\u03c0R}$\n\nD: $\\frac{\u03c0R}{180}$",
    "Given the rest of reaction components:\nreactant: Cc1ccc2c(cnn2C2CCCCO2)c1[B-](F)(F)F.[K+]\nligand: C1=C[C@H]2[Fe][C@@H]1[C]2P(c1ccccc1)c1ccccc1\nsolvent: CN(C)C=O\nbase: CCN(CC)CC  \nReactants list for selection:\nIc1ccc2ncccc2c1,Brc1ccc2ncccc2c1,Clc1ccc2ncccc2c1\nWhat is the optimal reactant?",
    "Let $a, b \\in \\mathbb{R}$. Then, \"$a + b > 4$\" is a (\u3000\u3000) condition for \"$a > 1$ and $b > 3$\".\nA: Sufficient but not necessary\nB: Necessary but not sufficient\nC: Necessary and sufficient\nD: Neither sufficient nor necessary",
    "Given the rest of reaction components:\nreactant: Cc1ccc2c(cnn2C2CCCCO2)c1[B-](F)(F)F.[K+]\nligand: CCCCP(C12CC3CC(CC(C3)C1)C2)C12CC3CC(CC(C3)C1)C2\nsolvent: CO\nbase: C(=O)(O)[O-].[Na+]  \nReactants list for selection:\nIc1ccc2ncccc2c1,Brc1ccc2ncccc2c1,Clc1ccc2ncccc2c1\nWhat is the optimal reactant?",
    "Why are water molecules attracted to other water molecules?\n\nA. Water molecules repel due to their neutrality.\nB. Water molecules are attracted by hydrogen bonds.\nC. Water molecules are attracted due to their non-polarity.\nD. Water molecules are attracted due to their polarity.",
    "Why is alternating current used to distribute electricity?\n\nA. AC is cheaper to produce than DC.\nB. AC is safer for long-distance transmission.\nC. AC allows voltage transformation, reducing power loss during transmission.\nD. AC can be easily converted to DC.",
    "Prove that if $\\alpha, \\beta, \\gamma$ are the angles of a triangle, then the following inequality holds:\n\n$$\n\\frac{\\cos \\alpha}{\\sin \\beta \\sin \\gamma}+\\frac{\\cos \\beta}{\\sin \\gamma \\sin \\alpha}+\\frac{\\cos \\gamma}{\\sin \\alpha \\sin \\beta} \\leq 3\n$$",
    "A 63-year-old man presents to his primary care physician for follow-up. He reports a slow and steady weight gain of 6 pounds over the past 6 months, despite attempts to control his diet and increase his level of exercise. His medications include pravastatin, lisinopril, and hydrochlorothiazide. On exam, his vital signs are stable. He is obese (BMI 32), and his waist circumference is 43 inches. His physician is concerned about an abnormal fasting blood glucose and dyslipidemia. On further work-up with oral glucose tolerance test, the patient is diagnosed with diabetes. Which of the following associations is consistent with this patient\u2019s most likely form of diabetes?\n\nA. Strong HLA class II gene makeup\nB. Pancreatic islet cell amyloid deposition\nC. Pancreatic islet cell leukocyte infiltration\nD. Auto-antibodies against pancreatic islet cell antigens",
    "A 42-year-old male presents to his primary care physician complaining of fatigue. He has not been to the doctor since he was 22 years of age. He reports that over the past three months, he has felt tired and weak despite no changes in diet or exercise. He is otherwise healthy and takes no medications. Family history is notable for colorectal cancer in his father and paternal uncle, ovarian cancer in his paternal grandmother, and pancreatic cancer in his paternal uncle. Physical examination is notable for conjunctival pallor. A complete blood count reveals a hemoglobin of 9.1 g/dL and hematocrit of 31%. A stool sample is hemoccult positive and a colonoscopy reveals a fungating hemorrhagic mass in the ascending colon. Which of the following processes is most likely impaired in this patient?\n\nA. Base excision repair\nB. Nucleotide excision repair\nC. Mismatch repair\nD. Non-homologous end joining",
    "Given the rest of reaction components:\nreactant: Cc1ccc2c(cnn2C2CCCCO2)c1B1OC(C)(C)C(C)(C)O1\nligand: C1=C[C@H]2[Fe][C@@H]1[C]2P(c1ccccc1)c1ccccc1\nsolvent: CO\nbase: [F-].[Cs+]  \nReactants list for selection:\nIc1ccc2ncccc2c1,Brc1ccc2ncccc2c1,Clc1ccc2ncccc2c1\nWhat is the optimal reactant?",
    "Jia walks from home to Yi's house. At the same time, Yi rides a bicycle from home towards Jia's house. Both maintain constant speeds, and Yi's riding speed is 5 times Jia's walking speed. The distance between their houses is 10,560 feet, and Jia's step length is 2.5 feet. How many steps has Jia taken when he meets Yi?\n(A) 704\n(B) 845\n(C) 1056\n(D) 1760\n(E) 3520",
    "Why are muscle cells needed in the intestine?\n\nA. Muscles absorb nutrients from food in the intestine.\nB. Muscles break down food particles in the intestine.\nC. Muscles protect the intestine from harmful substances.\nD. Muscles enable peristalsis for food transportation in the intestine.",
    "Is there an odd number \\( n \\geq 3 \\) and \\( n \\) distinct prime numbers \\( p_{1}, p_{2}, \\cdots, p_{n} \\) such that \\( p_{i}+p_{i+1} \\) (for \\( i=1,2, \\cdots, n \\) and \\( p_{n+1}=p_{1} \\)) are all perfect squares? Prove your conclusion.\n(11th Western China Mathematical Olympiad)\n\nIf \\( n \\) distinct prime numbers \\( a_{1}, a_{2}, \\cdots, a_{n} \\) are placed at the vertices of a convex \\( n \\)-gon \\( A_{1} A_{2} \\cdots A_{n} \\) such that the sum of the numbers at the endpoints of each side \\( a_{1}+a_{2}, a_{2}+a_{3}, \\cdots, a_{n-1}+a_{n}, a_{n}+a_{1} \\) are all perfect squares, this \\( n \\)-gon \\( A_{1} A_{2} \\cdots A_{n} \\) is called a \"high-quality \\( n \\)-gon.\" If two high-quality \\( n \\)-gons \\( A_{1} A_{2} \\cdots A_{n} \\) and \\( B_{1} B_{2} \\cdots B_{n} \\) have a set of \\( 2n \\) distinct prime numbers \\( a_{1}, a_{2}, \\cdots, a_{n}, b_{1}, b_{2}, \\cdots, b_{n} \\) such that \\( a_{1}+a_{2}=b_{1}+b_{2}, \\cdots, a_{n-1}+a_{n}=b_{n-1}+b_{n}, a_{n}+a_{1}=b_{n}+b_{1} \\), then the two \\( n \\)-gons are considered equal. Determine:\n(1) Do there exist two equal high-quality triangles?\n(2) Do there exist two equal high-quality quadrilaterals?\nProve your conclusions.",
    "What is compartmentalization of a cell?\n\nA. Cell compartmentalization\nB. Cell communication\nC. Cell division\nD. Cell fusion",
    "Points $P$ and $C$ are fixed on a circle; points $A$ and $B$ move along the circle such that the angle $A C B$ remains constant. Prove that the Simson lines of point $P$ with respect to the triangles $A B C$ are tangent to a fixed circle.",
    "How does water impact life processes?\n\nA. Water helps dissolve compounds, maintain osmolarity, regulate temperature, and support chemical reactions in cells.\nB. Water transports nutrients, provides buoyancy, prevents dehydration, and aids in waste removal.\nC. Water enhances vision, promotes growth, enables movement, and protects against pathogens.\nD. Water generates energy, facilitates communication, maintains pressure, and supports structural integrity.",
    "How do energy and power relate to work?\n\nA. \u0394E = P \u00d7 F \u00d7 s\nB. \u0394E = F \u00d7 s = P \u00d7 t\nC. \u0394E = P \u00d7 F \u00d7 t\nD. \u0394E = F \u00d7 t = P \u00d7 s",
    "What peaks in their mass spectra could be used to distinguish between 4-methyl-2-pentanone and 2-methyl-3-pentanone?\n\nA. M/z 57 and m/z 85\nB. M/z 43 and m/z 71\nC. M/z 29 and m/z 99\nD. M/z 58 and m/z 86",
    "Two semicircles, each with radius \\(\\sqrt{2}\\), are tangent to each other. If \\( AB \\parallel CD \\), determine the length of segment \\( AD \\).",
    "What is the difference between note and tone of a sound?\n\nA. Note: duration; Tone: loudness\nB. Note: pitch with a letter; Tone: sound quality\nC. Note: loudness; Tone: duration\nD. Note: sound quality; Tone: pitch with a letter",
    "If an object with a mass of $ 3kg  $ changes speed from $4m/s$ to $ 9m/s$, by how much does its kinetic energy change?\n\nA. 82.5 J\nB. 112.5 J\nC. 97.5 J\nD. 76.5 J",
    "Given the complex number $z= \\frac {1}{i(i+1)}$, then $|z|=$ \uff08\u3000\u3000\uff09\n\nA:  $\\frac { \\sqrt {2}}{2}$\n\nB:  $\\frac {1}{2}$\n\nC:  $\\frac {1}{4}$\n\nD:  $\\frac {1}{8}$",
    "The sequence $\\left\\{x_{n}\\right\\}$ is defined as follows: \\(x_{1}=1\\), \\(x_{2}=a\\), where \\(a\\) is a given real number greater than 1. If for a certain integer \\(n \\geqslant 1\\), the first \\(2^{n}\\) terms of the sequence have been defined, then for \\(2^{n}+1 \\leq k \\leq 2^{n+1}\\), define \\(x_{k}=a x_{k-2^{n}}\\). Therefore, the terms of the sequence \\(\\left\\{x_{n}\\right\\}\\) are \\(1, a, a, a^{2}, a, a^{2}, a^{2}, a^{3}, \\cdots\\). Let \\(S_{n}\\) denote the sum of the first \\(n\\) terms of the sequence. Prove that if the binary representation of the number \\(n\\) is\n$$\nn=2^{k_{0}}+2^{k_{1}}+\\cdots+2^{k_{r}}\\left(k_{0}>k_{1}>\\cdots>k_{r} \\geqslant 0\\right),\n$$\n\nthen \\[S_{n}=\\sum_{j=0}^{r} a^{j}(1+a)^{k_{j}}.\\]",
    "Given circle $C$: $x^{2}+y^{2}=3$, line $l$: $x+3y-6=0$, point $P(x_{0},y_{0})\u2208l$, there exists point $Q\u2208C$, such that $\u2220OPQ=60^{\\circ}(O$ is the coordinate origin$)$, determine the range of $x_{0}$.\nA: $\\[- \\frac {1}{2},1\\]$,\nB: $\\[0,1\\]$,\nC: $\\[0, \\frac {6}{5}\\]$,\nD: $\\[ \\frac {1}{2}, \\frac {3}{2}\\]$,",
    "A 48-year-old man is brought to the emergency department for sudden onset of difficulty breathing 6 hours ago. For the past several months, he has had shortness of breath on exertion and while lying down on the bed, frequent headaches, and swelling of his feet. He does not take any medications despite being diagnosed with hypertension 10 years ago. His pulse is 90/min, respirations are 20/min, blood pressure is 150/110 mm Hg, and temperature is 37.0\u00b0C (98.6\u00b0F). Physical examination shows an overweight male in acute distress with audible wheezes. Crackles are heard bilaterally and are loudest at the lung bases. Which of the following findings on cardiac auscultation will most likely be present in this patient?\n\nA. Loud P2\nB. S3 gallop\nC. Absent S4\nD. A loud S1",
    "Due to the impact of the epidemic, the turnover of a supermarket is growing slowly. The turnover of the supermarket in January is $36$ thousand dollars, and in March it is $48$ thousand dollars. Let $x$ be the average monthly growth rate from January to March. Which of the following equations is correct?\n\nA: $36\\left(1-x\\right)^{2}=48$\n\nB: $36\\left(1+x\\right)^{2}=48$\n\nC: $36\\left(1-x\\right)^{2}=48-36$\n\nD: $48\\left(1-x\\right)^{2}=36$",
    "What are homodesmotic reactions?\n\nA. Reactions where carbon atoms change their hybridization state, causing a change in molecular shape.\nB. Reactions where carbon atoms rearrange to form different molecular structures with the same overall composition.\nC. Reactions with equal numbers of carbon atoms in the same hybridization state and matching groups, used to evaluate strain energy in rings.\nD. Reactions involving the breaking and forming of carbon-carbon double bonds.",
    "Why were Rutherford's students surprised by the results of the gold foil experiment?\n\nA. Rutherford's students expected the gold foil to disintegrate, but it remained intact, leading to the discovery of the electron cloud.\nB. Rutherford's students expected alpha particles to be absorbed by the gold foil, but they passed through, disproving the Thompson 'Plum Pudding' Model.\nC. Rutherford's students expected alpha particles to bounce off the gold foil, but most passed through, leading to the Rutherford 'Shell' Model of the atom.\nD. Rutherford's students expected the gold foil to change color, but it didn't, proving the existence of a dense positive nucleus.",
    "Why do some people argue for preserving habitats?\n\nA. Preserving habitats maintains biodiversity, which is important for the environment and human survival.\nB. Preserving habitats improves the aesthetic value of the environment.\nC. Preserving habitats provides more space for human development.\nD. Preserving habitats reduces the risk of natural disasters.",
    "Given an acute-angled triangle \\( \\triangle ABC \\) is inscribed in circle \\( \\odot O \\), with \\( AB < AC \\). The angle bisector of \\(\\angle BAC\\) intersects \\(BC\\) at \\(T\\). Let \\(M\\) be the midpoint of \\(AT\\). Point \\(P\\) inside \\( \\triangle ABC \\) satisfies \\(PB \\perp PC\\). A perpendicular is drawn from \\(P\\) to \\(AP\\), intersecting at points \\(D\\) and \\(E\\) (distinct from \\(P\\)), with conditions \\(BD = BP\\) and \\(CE = CP\\). Given that line \\(AO\\) bisects segment \\(DE\\), prove that line \\(AO\\) is tangent to the circumcircle of \\( \\triangle AMP \\).",
    "To validate a clinical diagnostic tool, used by emergency physicians (EPs), to diagnose the central cause of patients presenting with vertigo, and to determine interrater reliability of this tool.\nA convenience sample of adult patients presenting to a single academic ED with isolated vertigo (i.e. vertigo without other neurological deficits) was prospectively evaluated with STANDING (SponTAneousNystagmus, Direction, head Impulse test, standiNG) by five trained EPs. The first step focused on the presence of spontaneous nystagmus, the second on the direction of nystagmus, the third on head impulse test and the fourth on gait. The local standard practice, senior audiologist evaluation corroborated by neuroimaging when deemed appropriate, was considered the reference standard. Sensitivity and specificity of STANDING were calculated. On the first 30 patients, inter-observer agreement among EPs was also assessed.\nFive EPs with limited experience in nystagmus assessment volunteered to participate in the present study enrolling 98 patients. Their average evaluation time was 9.9 \u00b1 2.8\u2009min (range 6-17). Central acute vertigo was suspected in 16 (16.3%) patients. There were 13 true positives, three false positives, 81 true negatives and one false negative, with a high sensitivity (92.9%, 95% CI 70-100%) and specificity (96.4%, 95% CI 93-38%) for central acute vertigo according to senior audiologist evaluation. The Cohen's kappas of the first, second, third and fourth steps of the STANDING were 0.86, 0.93, 0.73 and 0.78, respectively. The whole test showed a good inter-observer agreement (k = 0.76, 95% CI 0.45-1).\n\nCan emergency physicians accurately and reliably assess acute vertigo in the emergency department?",
    "What is the electric current produced when a voltage of $15 V$ is applied to a circuit with a resistance of $12 Omega$?\n\nA. 1.5A\nB. 1.25A\nC. 2A\nD. 1.75A",
    "Emergency surgery is associated with poorer outcomes and higher mortality with recent studies suggesting the 30-day mortality to be 14-15%. The aim of this study was to analyse the 30-day mortality, age-related 30-day mortality and 1-year mortality following emergency laparotomy. We hope this will encourage prospective data collection, improvement of care and initiate strategies to establish best practice in this area.\nThis was a retrospective study of patients who underwent emergency laparotomy from June 2010 to May 2012. The primary end point of the study was 30-day mortality, age-related 30-day mortality and 1-year all-cause mortality.\n477 laparotomies were performed in 446 patients. 57% were aged<70 and 43% aged>70 years. 30-day mortality was 12, 4% in those aged<70 years and 22% in those>70 years (p<0.001). 1-year mortality was 25, 15% in those aged under 70 years and 38% in those aged>70 years (p<0.001).\n\n30-Day and 1-year mortality in emergency general surgery laparotomies: an area of concern and need for improvement?",
    "Given the rest of reaction components:\nreactant 1: Clc1ccc2ncccc2c1\nreactant 2: Cc1ccc2c(cnn2C2CCCCO2)c1B(O)O\nligand: CN(C)Cc1ccc(P(C(C)(C)C)C(C)(C)C)cc1\nbase: [OH-].[K+]  \nSolvent list for selection:\nC1CCOC1,CN(C)C=O,CO\nWhat is the optimal solvent?",
    "A 3-year-old girl is brought to the cardiologist because of sweating and respiratory distress while eating. She is at the 30th percentile for height and 15th percentile for weight. Echocardiography shows a defect in the membranous portion of the interventricular septum and a moderately decreased left ventricular ejection fraction. Physical examination is most likely to show which of the following findings?\n\nA. Systolic murmur that increases with hand clenching\nB. Systolic murmur that increases with forced exhalation against a closed glottis\nC. Diastolic murmur preceded by opening snap\nD. Continuous murmur that is loudest at the second heart sound",
    "Polymorphisms in the oestrogen receptor 1 (ESR1) and oestrogen receptor 2 (ESR2) genes are associated with intermediate or endpoint markers of cardiovascular disease and with the efficacy of postmenopausal hormone therapy (HT). Contradictory findings have been described in the past and the role of these genetics variants remains unclear.\nA cross-sectional study was carried out with 266 postmenopausal women, of whom 115 received oral HT (HT+) and 151 did not receive any HT (HT-). We analysed three single-nucleotide polymorphisms (SNPs) in ESR1 (rs1801132, rs7757956 and rs2813544) and two in ESR2 (rs3020450 and rs7154455) and derived haplotypes with three additional polymorphisms that had been previously investigated by our group (ESR1 rs2234693 and ESR2 rs1256049 and rs4986938).\nThe ESR1 rs2813544 polymorphism was associated with low-density lipoprotein cholesterol (LDL-C) in HT+ postmenopausal women (p\u2009=\u20090.044; pC\u2009=\u20090.388), while one ESR2 gene haplotype was associated with total cholesterol (T-chol) (p\u2009=\u20090.015; pC\u2009=\u20090.090) and LDL-C in HT+ postmenopausal women (p\u2009=\u20090.021; pC\u2009=\u20090.126).\n\nAre polymorphisms in oestrogen receptors genes associated with lipid levels in response to hormone therapy?",
    "A 62-year-old woman with a history of hypertension and type 2 diabetes mellitus comes to the physician for a routine health maintenance examination. She has smoked 1 pack of cigarettes daily for the last 15 years. Current medications include glyburide and amlodipine. The physician prescribes a medication that decreases the production of mevalonate. Which of the following changes to the serum is most likely to develop as an adverse effect of the prescribed drug?\n\nA. Increased creatine kinase concentration\nB. Decreased glucose concentration\nC. Increased triglyceride concentration\nD. Increased bradykinin concentration",
    "A 7-day-old infant boy presents to an emergency department due to poor feeding. His parents are recent immigrants to the United States. He was born in a traditional home birth and has never seen a medical provider. Mom had no prenatal care, has no medical issues, and is unvaccinated. The baby had been breastfeeding well until 24 hours ago when mom noticed he started having trouble latching. In the last 12 hours, he has completely refused to feed. He has had a decreased number of wet diapers and has stooled twice in the last 24 hours. His temperature is 98.6\u00b0F (37.0\u00b0C), pulse is 180/min, respirations are 52/min, and blood pressure is 70/50 mmHg. On exam, the infant has increased tone, a clenched jaw, no head lag, and clenched hands. Initial screening bloodwork is normal. What is the most likely organism causing this infant's presentation?\n\nA. Clostridium botulinum\nB. Clostridium tetani\nC. Group B streptococcus\nD. Listeria monocytogenes",
    "Given the rest of reaction components:\nreactant: Cc1ccc2c(cnn2C2CCCCO2)c1B(O)O\nligand: CC(C)c1cc(C(C)C)c(-c2ccccc2P(C2CCCCC2)C2CCCCC2)c(C(C)C)c1\nsolvent: CO\nbase: [OH-].[Na+]  \nReactants list for selection:\nBrc1ccc2ncccc2c1,O=S(=O)(Oc1cnc2ccccc2c1)C(F)(F)F,Ic1ccc2ncccc2c1\nWhat is the optimal reactant?",
    "A 3-year-old boy is brought to the clinic by his parents because he \u2018hasn\u2019t been himself lately\u2019 and reportedly gets tired very easily from his swimming classes in comparison to the other kids. He also \u2018can\u2019t catch his breath\u2019 at times. The mother also reports that he seems to be slightly shorter than other children his age. His temperature is 36.6\u00b0C (97.9\u00b0F), blood pressure is 110/70 mm Hg, and respiratory rate is 14/min. On auscultation, a localized harsh pansystolic murmur is heard over the left sternal border at the level of the 2nd\u20133rd intercostal space. The murmur becomes louder when the patient is asked to squat. An echocardiogram is performed. Which of the structures below gives rise to the defective structure that is causing this patient\u2019s symptoms?\n\nA. Endocardial cushion\nB. Infundibular septum\nC. 3rd pharyngeal arch\nD. Rathke\u2019s pouch",
    "The cost of a phone call from location A to location B for $m$ minutes is given by $f(m) = 1.06(0.50 \\times [m] + 1)$, where $m > 0$, and $[m]$ is the smallest integer greater than or equal to $m$ (for example, $[3] = 3$, $[3.7] = 4$, $[3.1] = 4$). What is the cost of a phone call lasting 5.5 minutes from location A to location B?\n\nA: 3.71  \nB: 3.97  \nC: 4.24  \nD: 4.77",
    "Given the rest of reaction components:\nreactant 1: Brc1ccc2ncccc2c1\nreactant 2: Cc1ccc2c(cnn2C2CCCCO2)c1B(O)O\nligand: Cc1ccccc1P(c1ccccc1C)c1ccccc1C\nbase: [O-]P(=O)([O-])[O-].[K+].[K+].[K+]  \nSolvent list for selection:\nC1CCOC1,CN(C)C=O,CO\nWhat is the optimal solvent?",
    "Given the rest of reaction components:\nreactant: Cc1ccc2c(cnn2C2CCCCO2)c1B1OC(C)(C)C(C)(C)O1\nligand: c1ccc(P(c2ccccc2)c2ccccc2)cc1\nsolvent: CO\nbase: C(=O)(O)[O-].[Na+]  \nReactants list for selection:\nIc1ccc2ncccc2c1,Brc1ccc2ncccc2c1,Clc1ccc2ncccc2c1\nWhat is the optimal reactant?",
    "What is the angular momentum of an object with a mass of $5  kg$ that moves along a circular path of radius $8    m$ at a frequency of $ 6    Hz $?\n\nA. 24127.4 kgm^2s^-1\nB. 18159.2 kgm^2s^-1\nC. 6053.1 kgm^2s^-1\nD. 12063.7 kgm^2s^-1",
    "At a consultation, there were 20 students and 20 problems discussed. It turned out that each student solved two problems, and each problem was solved by two students. Prove that it is possible to organize the discussion of the problems in such a way that each student presents one of the problems they solved, and all problems are covered.",
    "A 38-year-old woman presents to the office for a routine examination. She has no complaints and offers very little information voluntarily. She answers each question with soft, short sentences and avoids eye contact. She appears timid, anxious and admits that this is one of the very few times she has left the house in the last several years. Medical history is significant for essential hypertension. She takes hydrochlorothiazide and a daily vitamin. She has worked from home as a web graphic designer for 20 years. Questions about her social life reveal that she is very shy with few friends, and she often makes excuses to avoid parties and social gatherings. Despite this, she expresses a strong desire to be in a relationship. Today, her blood pressure is 125/85 mm Hg, heart rate is 95/min, respiratory rate is 18/min, and temperature is 37.0\u00b0C (98.6\u00b0F). On physical examination, her heart has a regular rhythm and her lungs are clear to auscultation bilaterally. Which of the following is most consistent with her behavior?\n\nA. Schizoid personality disorder\nB. Avoidant personality disorder\nC. Antisocial personality disorder\nD. Agoraphobia"
] + ['Person A and person B each guess a riddle in each guessing activity. If one person guesses correctly and the other person guesses incorrectly, the person who guessed correctly wins; otherwise, it is a tie. It is known that in each activity, the probabilities of A and B guessing correctly are $\\frac{5}{6}$ and $\\frac{3}{5}$, respectively. The outcomes of A and B guessing correctly or incorrectly do not affect each other within each activity, and each activity is independent of others. The probability of A winning in one activity is ______; the probability of A winning at least 2 out of 3 activities is ______.',
 'Prove that if $\\alpha, \\beta, \\gamma$ are the angles of a triangle, then the following inequality holds:\n\n$$\n\\frac{\\cos \\alpha}{\\sin \\beta \\sin \\gamma}+\\frac{\\cos \\beta}{\\sin \\gamma \\sin \\alpha}+\\frac{\\cos \\gamma}{\\sin \\alpha \\sin \\beta} \\leq 3\n$$',
 'The graph of the function $y=\\cos x - \\sin x$ has a line of symmetry given by (__).\n\nA: $x= \\frac{\\pi}{4}$\nB: $x= \\frac{\\pi}{8}$\nC: $x= -\\frac{\\pi}{8}$\nD: $x= -\\frac{\\pi}{4}$',
 'Let $a, b \\in \\mathbb{R}$. Then, "$a + b > 4$" is a (\u3000\u3000) condition for "$a > 1$ and $b > 3$".\nA: Sufficient but not necessary\nB: Necessary but not sufficient\nC: Necessary and sufficient\nD: Neither sufficient nor necessary',
 'To assess and compare the value of split-liver transplantation (SLT) and living-related liver transplantation (LRT).\nThe concept of SLT results from the development of reduced-size transplantation. A further development of SLT, the in situ split technique, is derived from LRT, which itself marks the optimized outcome in terms of postoperative graft function and survival. The combination of SLT and LRT has abolished deaths on the waiting list, thus raising the question whether living donor liver transplantation is still necessary.\nOutcomes and postoperative liver function of 43 primary LRT patients were compared with those of 49 primary SLT patients (14 ex situ, 35 in situ) with known graft weight performed between April 1996 and December 2000. Survival rates were analyzed using the Kaplan-Meier method.\nAfter a median follow-up of 35 months, actual patient survival rates were 82% in the SLT group and 88% in the LRT group. Actual graft survival rates were 76% and 81%, respectively. The incidence of primary nonfunction was 12% in the SLT group and 2.3% in the LRT group. Liver function parameters (prothrombin time, factor V, bilirubin clearance) and surgical complication rates did not differ significantly. In the SLT group, mean cold ischemic time was longer than in the LRT group. Serum values of alanine aminotransferase during the first postoperative week were significantly higher in the SLT group. In the LRT group, there were more grafts with signs of fatty degeneration than in the SLT group.\n\nIs there still a need for living-related liver transplantation in children?',
 'Is there an odd number \\( n \\geq 3 \\) and \\( n \\) distinct prime numbers \\( p_{1}, p_{2}, \\cdots, p_{n} \\) such that \\( p_{i}+p_{i+1} \\) (for \\( i=1,2, \\cdots, n \\) and \\( p_{n+1}=p_{1} \\)) are all perfect squares? Prove your conclusion.\n(11th Western China Mathematical Olympiad)\n\nIf \\( n \\) distinct prime numbers \\( a_{1}, a_{2}, \\cdots, a_{n} \\) are placed at the vertices of a convex \\( n \\)-gon \\( A_{1} A_{2} \\cdots A_{n} \\) such that the sum of the numbers at the endpoints of each side \\( a_{1}+a_{2}, a_{2}+a_{3}, \\cdots, a_{n-1}+a_{n}, a_{n}+a_{1} \\) are all perfect squares, this \\( n \\)-gon \\( A_{1} A_{2} \\cdots A_{n} \\) is called a "high-quality \\( n \\)-gon." If two high-quality \\( n \\)-gons \\( A_{1} A_{2} \\cdots A_{n} \\) and \\( B_{1} B_{2} \\cdots B_{n} \\) have a set of \\( 2n \\) distinct prime numbers \\( a_{1}, a_{2}, \\cdots, a_{n}, b_{1}, b_{2}, \\cdots, b_{n} \\) such that \\( a_{1}+a_{2}=b_{1}+b_{2}, \\cdots, a_{n-1}+a_{n}=b_{n-1}+b_{n}, a_{n}+a_{1}=b_{n}+b_{1} \\), then the two \\( n \\)-gons are considered equal. Determine:\n(1) Do there exist two equal high-quality triangles?\n(2) Do there exist two equal high-quality quadrilaterals?\nProve your conclusions.',
 'Two semicircles, each with radius \\(\\sqrt{2}\\), are tangent to each other. If \\( AB \\parallel CD \\), determine the length of segment \\( AD \\).',
 'If the degree of the central angle corresponding to an arc increases by $1^{\\circ}$, and the radius of the arc is $R$, then the arc length increases by ( )\n\nA: $\\frac{πR}{360}$\n\nB: $\\frac{180}{πR}$\n\nC: $\\frac{360}{πR}$\n\nD: $\\frac{πR}{180}$',
 'During the Northern and Southern Dynasties, there was a mathematical problem in the ancient book "Zhang Qiujian\'s Arithmetic": "There are ten ranks of people today. For each rank, the palace grants gold with a decreasing arithmetic sequence. The top three people received a total of 4 pounds of gold; the last four people each received 3 pounds of gold; and the remaining three people in the middle ranks will also receive gold according to the arithmetic sequence. Question: How many more pounds of gold does each rank receive than the rank below it?"\n\nA: $\\frac{4}{39}$\nB: $\\frac{7}{78}$\nC: $\\frac{7}{76}$\nD: $\\frac{5}{85}$',
 'Given the complex number $z= \\frac {1}{i(i+1)}$, then $|z|=$ （\u3000\u3000）\n\nA:  $\\frac { \\sqrt {2}}{2}$\n\nB:  $\\frac {1}{2}$\n\nC:  $\\frac {1}{4}$\n\nD:  $\\frac {1}{8}$',
 'Given that the three sides of triangle $\\triangle ABC$ are $a$, $b$, and $c$, which of the following conditions cannot determine that $\\triangle ABC$ is a right triangle?\n\nA: $\\angle A + \\angle B = \\angle C$\n\nB: $a^{2} = 5$, $b^{2} = 12$, $c^{2} = 13$\n\nC: $\\angle A : \\angle B : \\angle C = 1 : 1 : 2$\n\nD: $a = 7$, $b = 24$, $c = 25$',
 'A curious tourist wants to walk through the streets of the Old Town from the train station (point $A$ on the map) to their hotel (point $B$). The tourist wants their route to be as long as possible, but they are not interested in visiting the same intersection twice, so they do not do so. Draw the longest possible route on the map and prove that there is no longer route.',
 'Given an acute-angled triangle \\( \\triangle ABC \\) is inscribed in circle \\( \\odot O \\), with \\( AB < AC \\). The angle bisector of \\(\\angle BAC\\) intersects \\(BC\\) at \\(T\\). Let \\(M\\) be the midpoint of \\(AT\\). Point \\(P\\) inside \\( \\triangle ABC \\) satisfies \\(PB \\perp PC\\). A perpendicular is drawn from \\(P\\) to \\(AP\\), intersecting at points \\(D\\) and \\(E\\) (distinct from \\(P\\)), with conditions \\(BD = BP\\) and \\(CE = CP\\). Given that line \\(AO\\) bisects segment \\(DE\\), prove that line \\(AO\\) is tangent to the circumcircle of \\( \\triangle AMP \\).',
 'The sequence $\\left\\{x_{n}\\right\\}$ is defined as follows: \\(x_{1}=1\\), \\(x_{2}=a\\), where \\(a\\) is a given real number greater than 1. If for a certain integer \\(n \\geqslant 1\\), the first \\(2^{n}\\) terms of the sequence have been defined, then for \\(2^{n}+1 \\leq k \\leq 2^{n+1}\\), define \\(x_{k}=a x_{k-2^{n}}\\). Therefore, the terms of the sequence \\(\\left\\{x_{n}\\right\\}\\) are \\(1, a, a, a^{2}, a, a^{2}, a^{2}, a^{3}, \\cdots\\). Let \\(S_{n}\\) denote the sum of the first \\(n\\) terms of the sequence. Prove that if the binary representation of the number \\(n\\) is\n$$\nn=2^{k_{0}}+2^{k_{1}}+\\cdots+2^{k_{r}}\\left(k_{0}>k_{1}>\\cdots>k_{r} \\geqslant 0\\right),\n$$\n\nthen \\[S_{n}=\\sum_{j=0}^{r} a^{j}(1+a)^{k_{j}}.\\]',
 'The cost of a phone call from location A to location B for $m$ minutes is given by $f(m) = 1.06(0.50 \\times [m] + 1)$, where $m > 0$, and $[m]$ is the smallest integer greater than or equal to $m$ (for example, $[3] = 3$, $[3.7] = 4$, $[3.1] = 4$). What is the cost of a phone call lasting 5.5 minutes from location A to location B?\n\nA: 3.71  \nB: 3.97  \nC: 4.24  \nD: 4.77',
 'At a consultation, there were 20 students and 20 problems discussed. It turned out that each student solved two problems, and each problem was solved by two students. Prove that it is possible to organize the discussion of the problems in such a way that each student presents one of the problems they solved, and all problems are covered.',
 'Due to the impact of the epidemic, the turnover of a supermarket is growing slowly. The turnover of the supermarket in January is $36$ thousand dollars, and in March it is $48$ thousand dollars. Let $x$ be the average monthly growth rate from January to March. Which of the following equations is correct?\n\nA: $36\\left(1-x\\right)^{2}=48$\n\nB: $36\\left(1+x\\right)^{2}=48$\n\nC: $36\\left(1-x\\right)^{2}=48-36$\n\nD: $48\\left(1-x\\right)^{2}=36$',
 'Given four lines such that any three of them determine a triangle, prove that the orthocenters of these four triangles are collinear.',
 'Points $P$ and $C$ are fixed on a circle; points $A$ and $B$ move along the circle such that the angle $A C B$ remains constant. Prove that the Simson lines of point $P$ with respect to the triangles $A B C$ are tangent to a fixed circle.',
 'Given circle $C$: $x^{2}+y^{2}=3$, line $l$: $x+3y-6=0$, point $P(x_{0},y_{0})∈l$, there exists point $Q∈C$, such that $∠OPQ=60^{\\circ}(O$ is the coordinate origin$)$, determine the range of $x_{0}$.\nA: $\\[- \\frac {1}{2},1\\]$,\nB: $\\[0,1\\]$,\nC: $\\[0, \\frac {6}{5}\\]$,\nD: $\\[ \\frac {1}{2}, \\frac {3}{2}\\]$,',
 "Jia walks from home to Yi's house. At the same time, Yi rides a bicycle from home towards Jia's house. Both maintain constant speeds, and Yi's riding speed is 5 times Jia's walking speed. The distance between their houses is 10,560 feet, and Jia's step length is 2.5 feet. How many steps has Jia taken when he meets Yi?\n(A) 704\n(B) 845\n(C) 1056\n(D) 1760\n(E) 3520"]

# All datasets have the same columns
DS_COLUMNS = {"question", "solution", "cot_type", "source_type", "metadata"}

### Load functions ###
def load_generic(name, split, question_field="question", solution_field="solution", cot_type="math", version_tag=None):
    all_questions = []
    all_examples = []
    conf = "gpqa_diamond" if name == "Idavidrein/gpqa" else None
    ds = datasets.load_dataset(name, conf, version_tag=version_tag, trust_remote_code=True)[split]
    # Make metadata a string that can be loaded via literal_eval to avoid TypeError: Couldn't cast array of type list<item: string> to null 
    ds = ds.map(lambda x: {"question": x.pop(question_field), "solution": x.pop(solution_field, None), "cot_type": cot_type, "source_type": name, "metadata": str(x)})
    ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS])
    return ds

def load_math():
    ds = datasets.load_dataset("simplescaling/openaimath", trust_remote_code=True)["train"]
    ds = ds.map(lambda x: {"question": x.pop("problem"), "solution": x.pop("solution"), "cot_type": "math", "source_type": "simplescaling/openaimath/" + x['subject'], "metadata": str(x)})
    ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS])
    return ds

# def load_numinamath():
#     ds = datasets.load_dataset("AI-MO/NuminaMath-CoT", trust_remote_code=True)["train"]
#     ds = ds.filter(lambda x: x["source"] in ["cn_k12", "aops_forum", "olympiads"])
#     ds = ds.map(lambda x: {"question": x.pop("problem"), "solution": x.pop("solution"), "cot_type": "math", "source_type": "AI-MO/NuminaMath-CoT/" + x["source"], "metadata": str(x)})
#     ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS])
#     return ds

def load_numinamath():
    ds = datasets.load_dataset("AI-MO/NuminaMath-CoT", trust_remote_code=True)["train"]
    ds_aops = ds.filter(lambda x: x["source"] == "aops_forum")
    ds = ds.filter(lambda x: x["source"] != "aops_forum")
    
    ### TMP ###
    questions = datasets.load_dataset("simplescaling/numinamath_500", trust_remote_code=True)["train"]['problem']
    ds = ds.filter(lambda x: (x["problem"] in questions) or (x["problem"] in TMP_QUESTIONS_TO_KEEP))
    ds = datasets.concatenate_datasets([ds, ds_aops])
    
    ds = ds.map(lambda x: {"question": x.pop("problem"), "solution": x.pop("solution"), "cot_type": "math", "source_type": "AI-MO/NuminaMath-CoT/" + x["source"], "metadata": str(x)})
    ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS])
    return ds

def load_olympic_arena():
    confs = ['Math', 'Physics', 'Chemistry', 'Biology', 'Geography', 'Astronomy', 'CS']
    subject_to_o1domain = {"Math": "math", "CS": "coding"}
    ds = [datasets.load_dataset("GAIR/OlympicArena", c, trust_remote_code=True) for c in confs]
    ds = datasets.concatenate_datasets([d['test'] for d in ds] + [d['val'] for d in ds])
    # Filter for EN & text-only
    ds = ds.filter(lambda x: (x["language"] == "EN") and (x["modality"] == "text-only"))
    ds = ds.map(lambda x: {"question": x.pop("problem"), "solution": x.pop("solution"), "cot_type": subject_to_o1domain.get(x['subject'], "science"), "source_type": "GAIR/OlympicArena/" + x['subject'], "metadata": str(x)})
    ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS])
    return ds

def load_theoremqa():
    ds = datasets.load_dataset("TIGER-Lab/TheoremQA", trust_remote_code=True)["test"]
    ds = ds.filter(lambda x: x["Picture"] is None)
    ds = ds.map(lambda x: {"question": x.pop("Question"), "solution": x.pop("Answer"), "cot_type": "math", "source_type": "TIGER-Lab/TheoremQA/" + x['Answer_type'], "metadata": str(x)})
    ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS])
    return ds

def load_scieval():
    """
    Category Physics; Task: SocraticQA
    What is the moment of inertia of a pendulum with a mass of $2 kg$ that is $7  m$ from the pivot?\n\nA. 56 kgm^2\nB. 196 kgm^2\nC. 84 kgm^2\nD. 98 kgm^2\n\nAnswer:

    Category Chemistry; Task: SocraticQA
    What is the molecular geometry of the $PF_3$ molecule?\n\nA. Trigonal planar\nB. Bent\nC. Trigonal pyramidal\nD. Tetrahedral\n\nAnswer:
    Category Chemistry; Task: reagent selection
    Given the rest of reaction components:\nreactant 1: Ic1ccc2ncccc2c1\nreactant 2: Cc1ccc2c(cnn2C2CCCCO2)c1B1OC(C)(C)C(C)(C)O1\nligand: c1ccc(P(c2ccccc2)c2ccccc2)cc1\nbase: C(=O)(O)[O-].[Na+]  \nSolvent list for selection:\nC1CCOC1,CN(C)C=O,CO\nOptimal solvent:

    Category Biology; Task: MedQA
    A 74-year-old man was admitted to the intensive care ward due to progressive dyspnea, cough with pink sputum, and diaphoresis. He had 2 myocardial infarctions at the age of 66 and 69 years and suffers from chronic heart failure. At the time of presentation, his vital signs are as follows: blood pressure 90/50 mm Hg, heart rate 108/min, respiratory rate 29/min, and temperature 35.5°C (95.9°F). On physical examination, the patient sits upright. He is lethargic and cyanotic. Lung auscultation reveals widespread bilateral fine rales. Cardiac examination is significant for S3, accentuation of the pulmonic component of S2, and a systolic murmur heard best at the apex of the heart. Soon after hospitalization, the patient develops ventricular fibrillation and dies despite adequate resuscitation measures. Which microscopic finding would you expect to see in this patient on autopsy?\n\nA. Brownish inclusions in the pulmonary macrophages on H&E staining\nB. Positive Prussian-blue staining of the kidney tissue\nC. Ground-glass hepatocytes\nD. Positive Congo-red staining of the cardiac tissue\n\nAnswer:
    Category Biology; Task: PubMedQA
    Polymorphisms in the oestrogen receptor 1 (ESR1) and oestrogen receptor 2 (ESR2) genes are associated with intermediate or endpoint markers of cardiovascular disease and with the efficacy of postmenopausal hormone therapy (HT). Contradictory findings have been described in the past and the role of these genetics variants remains unclear.\nA cross-sectional study was carried out with 266 postmenopausal women, of whom 115 received oral HT (HT+) and 151 did not receive any HT (HT-). We analysed three single-nucleotide polymorphisms (SNPs) in ESR1 (rs1801132, rs7757956 and rs2813544) and two in ESR2 (rs3020450 and rs7154455) and derived haplotypes with three additional polymorphisms that had been previously investigated by our group (ESR1 rs2234693 and ESR2 rs1256049 and rs4986938).\nThe ESR1 rs2813544 polymorphism was associated with low-density lipoprotein cholesterol (LDL-C) in HT+ postmenopausal women (p\u2009=\u20090.044; pC\u2009=\u20090.388), while one ESR2 gene haplotype was associated with total cholesterol (T-chol) (p\u2009=\u20090.015; pC\u2009=\u20090.090) and LDL-C in HT+ postmenopausal women (p\u2009=\u20090.021; pC\u2009=\u20090.126).\n\nAre polymorphisms in oestrogen receptors genes associated with lipid levels in response to hormone therapy?\n\nAnswer:
    Category Biology; Task: SocraticQA
    What substance is transported across the inner membrane of the mitochondria?\n\nA. Glucose\nB. Protons\nC. Oxygen\nD. Electrons\n\nAnswer:
    """
    ds = datasets.load_dataset("OpenDFM/SciEval", trust_remote_code=True)
    ds = datasets.concatenate_datasets([ds['test'], ds['validation']])
    # As there's enough samples, filter out ones without answer (13011 out of 27533)
    ds = ds.filter(lambda x: x["answer"] is not None)
    # Remove the "\n\nAnswer:"; Replace "\nOptimal solvent:" with "\nWhat is the optimal solvent?"; "\nOptimal ligand:" with "\nWhat is the optimal ligand?"; "\nOptimal reactant" with "\nWhat is the optimal reactant?"
    def clean_question(x):
        x["question"] = x["question"].split("\n\nAnswer:")[0].replace("\nOptimal solvent:", "\nWhat is the optimal solvent?").replace("\nOptimal ligand:", "\nWhat is the optimal ligand?").replace("\nOptimal reactant:", "\nWhat is the optimal reactant?")
        return x
    ds = ds.map(clean_question)
    task_to_o1domain = {"SocraticQA": "science", "reagent selection": "science", "MedQA": "health science", "PubMedQA": "health science"}
    ds = ds.map(lambda x: {"question": x.pop("question"), "solution": x.pop("answer")[0], "cot_type": task_to_o1domain[x['task_name']], "source_type": "OpenDFM/SciEval/" + x['category'] + "/" + x['type'] + "/" + x['task_name'], "metadata": str(x)})
    ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS])
    return ds

def load_olympiad_bench():
    # Only EN & TO (text-only); Both OE (open-ended) and TP (Theorem proof)
    confs = ["OE_TO_maths_en_COMP", "OE_TO_physics_en_COMP", "TP_TO_maths_en_COMP", "TP_TO_physics_en_COMP"]
    # Multimodal: "OE_MM_maths_en_COMP", "OE_MM_physics_en_COMP", "TP_MM_maths_en_COMP", "TP_MM_physics_en_COMP"
    ds = [datasets.load_dataset("Hothan/OlympiadBench", c, trust_remote_code=True)['train'] for c in confs]
    ds = datasets.concatenate_datasets(ds)
    ### TODO: Is solution ever longer than 1?
    ### TODO: forgot to add context to question
    # The physics one is also rather math-heavy
    ds = ds.map(lambda x: {"question": x.pop("question"), "solution": x.pop("solution")[0], "cot_type": "math", "source_type": "Hothan/OlympiadBench/" + x['question_type'] + "/" + x['subject'], "metadata": str(x)})
    ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS])
    return ds

def load_jeebench():
    ds = datasets.load_dataset("daman1209arora/jeebench", trust_remote_code=True)['test']
    subject_to_o1domain = {"math": "math", "phy": "math", "chem": "science"}
    ds = ds.map(lambda x: {"question": x.pop("question"), "solution": x.pop("gold"), "cot_type": subject_to_o1domain[x['subject']], "source_type": "daman1209arora/jeebench/" + x['subject'], "metadata": str(x)})
    ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS])
    return ds

def load_agieval():
    conf_to_o1domain = {"sat_en": "english", "sat_math": "math", "lsat_ar": "english", "lsat_lr": "english", "lsat_rc": "english", "logiqa": "math"}
    confs = ["sat_en", "sat_math", 'lsat_ar', 'lsat_lr', 'lsat_rc', 'logiqa']
    # Some have empty passages so strip in that case; 'options' field needed even for sat_math as sometimes "Which of the following" in question
    ds_with_passage = [
        datasets.load_dataset("baber/agieval", c, trust_remote_code=True).map(lambda x: {"question": (x.pop("passage") + "\n\n").strip() + x.pop("question") + "\n" + "\n".join(x.pop("options")), "solution": x.pop("solution"), "cot_type": conf_to_o1domain[c] ,"source_type": "baber/agieval/" + c, "metadata": str(x)})
        for c in confs
    ]

    confs = ['aqua_rat']
    ds_no_passage = [
        datasets.load_dataset("baber/agieval", c, trust_remote_code=True).map(lambda x: {"question": x.pop("question"), "solution": x.pop("solution"), "cot_type": "math", "source_type": "baber/agieval/" + c, "metadata": str(x)})
        for c in confs
    ]

    # Only take most difficult questions; 'options' field not really needed
    ds = datasets.load_dataset("baber/agieval", 'math_agieval', trust_remote_code=True).filter(lambda x: x['level'] == 5).map(lambda x: {"question": x.pop("question"), "solution": x.pop("solution"), "cot_type": "math", "source_type": "baber/agieval/math_agieval", "metadata": str(x)})

    ds = [datasets.concatenate_datasets([d['test'], d['few_shot']]) for d in ds_with_passage + ds_no_passage + [ds]]
    ds = [d.remove_columns([c for c in d.column_names if c not in DS_COLUMNS]) for d in ds]
    ds = datasets.concatenate_datasets(ds)
    return ds

def load_statsqual():
    ds = datasets.load_dataset("simplescaling/s1-prob", trust_remote_code=True)['train']
    ds = ds.map(lambda x: {"question": x.pop("question"), "solution": x.pop("solution"), "cot_type": "math", "source_type": "simplescaling/s1-prob", "metadata": str(x)})
    ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS])
    return ds

def load_gpqa_extended():
    gpqa_to_o1domain = {"Chemistry": "science", "Biology": "science", "Physics": "science"}
    ds = datasets.load_dataset("Idavidrein/gpqa", "gpqa_extended", trust_remote_code=True)['train']
    # Filter against diamond
    ds_diamond = datasets.load_dataset("Idavidrein/gpqa", "gpqa_diamond", trust_remote_code=True)['train']
    ds = ds.filter(lambda x: x["Question"] not in ds_diamond["Question"])
    ds = ds.map(lambda x: {"question": x.pop("Question"), "solution": x.pop("Explanation"), "cot_type": gpqa_to_o1domain[x['High-level domain']], "source_type": "Idavidrein/gpqa", "metadata": str(x)})

    ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS])
    return ds
    
def load_xword():
    ds = datasets.load_dataset("0xharib/xword1", trust_remote_code=True)['train']

    # Use slightly different format, e.g. would need to Rename instruction, input, output -> cl
    # ds2 = datasets.load_dataset("0xharib/xword2", trust_remote_code=True)['train']
    # ds3 = datasets.load_dataset("0xharib/xword3", trust_remote_code=True)['train']
    # ds = datasets.concatenate_datasets([ds1, ds2, ds3])

    instruction = "Solve the crossword puzzle. You are presented with a clue as input and the number of letters in brackets."
    ds = ds.map(lambda x: {"question": instruction + "\n\n" + x.pop("input").split("### Clue: ")[1], "solution": x.pop("output"), "cot_type": "crossword", "source_type": "0xharib/xword1", "metadata": str(x)})
    ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS])
    return ds

def load_usaco():
    ds = datasets.load_dataset("codegenning/usacobench_formatted")['test']
    ds = ds.map(lambda x: {"question": x.pop("question").strip(), "solution": None, "cot_type": "coding", "source_type": "codegenning/usacobench_formatted", "metadata": str(x)})
    ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS])
    return ds

def load_quant():
    ds = datasets.load_dataset("simplescaling/s1-teasers")['train']
    ds = ds.map(lambda x: {"question": x.pop("Question").strip(), "solution": x.pop("Answer"), "cot_type": "math", "source_type": "simplescaling/s1-teasers", "metadata": str(x)})
    ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS])
    return ds

def load_livecodebench():
    versions = ["release_v1", "release_v2", "release_v3"]
    datasets_list = []
    for version in versions:
        ds = datasets.load_dataset("livecodebench/code_generation_lite", version_tag=version, trust_remote_code=True)["test"]
        ds = ds.map(lambda x: {
                "question": x.pop("question_content").strip(), 
                "solution": None, 
                "cot_type": "coding", 
                "source_type": f"LiveCodeBench/{version}", 
                "metadata": str(x)
            })
        # filter only the difficult questions
        ds = ds.filter(lambda x: x["difficulty"] == "hard")
        ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS])
        datasets_list.append(ds)
      
    final_ds = datasets.concatenate_datasets(datasets_list)
    return final_ds


### Selection functions ###
def select_examples_omni_math(clean_examples, n_examples):
    ### TODO: Filter out BAD_OMNIMATH_SAMPLES
    random.shuffle(clean_examples)
    all_subdomains = list()
    all_difficulties = list()
    # get all subdomains and difficulties
    for ex in tqdm(clean_examples, desc="Selecting examples"):
        meta = eval(ex["metadata"])
        try:
            domain = meta["domain"][0].split(" -> ")
        except:
            print("not domain found")
            clean_examples.remove(ex)
            continue
        try:
            subdomain = domain[2]
        except:
            print("not subdomain found")
            clean_examples.remove(ex)
            continue
        if len(subdomain) > 30:
            continue
        difficulty = meta["difficulty"]
        all_subdomains.append(subdomain)
        all_difficulties.append(difficulty)
    
    # make a counter for each subdomain and difficulty
    subdomain_counter = Counter(all_subdomains)
    difficulty_counter = Counter(all_difficulties)

    # sort in descending order of subdomain count
    subdomain_counter = sorted(subdomain_counter.items(), key=lambda x: x[1], reverse=False)
    print("subdomain_counter: ", len(subdomain_counter))
    # select n_examples from clean_examples covering diverse domains (select examples from each subdomain) and hard questions (difficulty > 6)
    selected_examples = []
    selected_subdomains = []
    selected_difficulties = []
    for (subdomain, _) in subdomain_counter:
        count = 0    
        for ex in clean_examples:
            meta = eval(ex["metadata"])
            if meta["domain"][0].split(" -> ")[2] == subdomain and meta["difficulty"] > 6.5:
                selected_examples.append(ex)
                selected_subdomains.append(subdomain)
                selected_difficulties.append(meta["difficulty"])
                count += 1
                if len(selected_examples) == n_examples:
                    return selected_examples, selected_subdomains, selected_difficulties
                # no more than 50 examples per subdomain
                # if count > 40: break
    return selected_examples, selected_subdomains, selected_difficulties

def select_examples_scieval(ds, n_examples):
    ### TMP ###
    samples_to_keep = ds.filter(lambda x: x['question'] in TMP_QUESTIONS_TO_KEEP)
    n_examples -= len(samples_to_keep)
    
    import math
    tasks = set([eval(x)['task_name'] for x in ds['metadata']])
    tasks_to_num_samples = {task: math.ceil(n_examples/len(tasks)) for task in tasks}
    # Only SocraticQA has topics
    topics = set([eval(x)['topic'] for x in ds['metadata']])
    socratic_qa_samples = tasks_to_num_samples["SocraticQA"]
    topics_to_num_samples = {topic: math.ceil(socratic_qa_samples/len(topics)) for topic in topics}
    selected_examples = []
    ds = ds.shuffle(seed=42)
    for i, ex in enumerate(ds):
        meta = eval(ex['metadata'])
        if meta['topic'] in topics: 
            if topics_to_num_samples[meta['topic']] > 0:
                selected_examples.append(ex)
                topics_to_num_samples[meta['topic']] -= 1
                tasks_to_num_samples["SocraticQA"] -= 1
        if (meta['task_name'] in tasks) and (tasks_to_num_samples[meta['task_name']] > 0):
            selected_examples.append(ex)
            tasks_to_num_samples[meta['task_name']] -= 1
        if len(selected_examples) == n_examples: break

    ### TMP ###
    ds = datasets.concatenate_datasets([datasets.Dataset.from_list(selected_examples), samples_to_keep])
    return ds

    

def decontaminate_train_data(train_questions, test_questions, ds, ngram_size=8):    
    # Build ngram lookups
    train_lookup = build_ngram_lookup(train_questions, ngram_size)
    test_lookup = build_ngram_lookup(test_questions, ngram_size)

    # Find contaminated questions
    contaminated_ids = find_contaminated_questions(train_lookup, test_lookup)

    # Remove contaminated examples
    not_contaminated_ids = set(range(len(train_questions))) - contaminated_ids
    ds = ds.select(list(not_contaminated_ids))
    print(f"\nDecontamination Results:")
    print(f"Total train questions: {len(train_questions)}")
    print(f"Contaminated questions: {len(contaminated_ids)}")
    print(f"Contamination rate: {(len(contaminated_ids)/len(train_questions)*100):.2f}%")
    print(f"Clean examples remaining: {len(ds)}")
    return ds

DS_TO_SELECTION = {
    # Name: [load function, selection function, #samples]

    # Very high-quality so take all (12K)
    "MATH": [load_math, None, None],
    # Very high-quality so take all (3922)
    "OlympicArena": [load_olympic_arena, None, None],
    # Take all (720)
    "TheoremQA": [load_theoremqa, None, None],
    # Pretty big (434,778) and unclear how high quality so take 500
    "NuminaMath": [load_numinamath, None, None],
    # Take all as super high-quality (3329)
    # "Omni-MATH": [partial(load_generic, name="KbsdJames/Omni-MATH", question_field="problem", split="test"), select_examples_omni_math, None],
    "Omni-MATH": [partial(load_generic, name="KbsdJames/Omni-MATH", question_field="problem", split="test"), None, None],
    # Not super diverse (only 4 subtasks which same question format) so do not take all (14228)
    "SciEval": [load_scieval, select_examples_scieval, 250],
    # Very high-quality so take all (626)
    "OlympiadBench": [load_olympiad_bench, None, None],
    # Very high-quality so take all (483)
    "JEEBench": [load_jeebench, None, None],
    # Very high-quality so take all
    "AGIEval": [load_agieval, None, None],
    # Very high-quality so take all (182)
    "StatsQual": [load_statsqual, None, None],
    # Very high-quality so take all
    "GPQA": [load_gpqa_extended, None, None],
    # High-quality but a lot so take 1000
    "XWord": [load_xword, None, 1000],
    # Very high-quality so take all (520)
    "USACO": [load_usaco, None, None],
    # Very high-quality so take all
    "Quant": [load_quant, None, None],
    # Very high-quality so take all
    "LiveCodeBench": [load_livecodebench, None, None],    
}

if __name__ == "__main__":
    random.seed(42)
    ### Load all ###
    # Load test questions
    test_datasets = {
        "AI-MO/aimo-validation-aime": {"split": "train", "question_field": "problem"},
        "Idavidrein/gpqa": {"split": "train", "question_field": "Question"},
        "simplescaling/openaimath": {"split": "test", "question_field": "problem"},
        "livecodebench/code_generation_lite": {"split": "test", "question_field": "question_content", "version_tag": "release_v4"},
    }
    test_questions = []
    for name, config in tqdm(test_datasets.items(), desc="Loading test questions"):
        test_questions.extend(load_generic(name, **config)['question'])

    ds_all = []
    for ds_name, (load_fn, selection_fn, n_samples) in DS_TO_SELECTION.items():
        print(f"Processing {ds_name}...")
        ds = load_fn()
        # ds = decontaminate_train_data(ds['question'], test_questions, ds, ngram_size=8)
        if selection_fn:
            # Outdated, needs to be updated
            if ds_name == "Omni-MATH":
                ds_list = list(ds)
                selected_examples, selected_subdomains, selected_difficulties = selection_fn(ds_list, n_samples)
                ds = datasets.Dataset.from_list(selected_examples)
            else:
                ds = selection_fn(ds, n_samples)
        else:
            ds = ds.shuffle(seed=42)
            if n_samples:
                questions_to_keep = ds.filter(lambda x: x['question'] in TMP_QUESTIONS_TO_KEEP)
                n_samples -= len(questions_to_keep)
                ds = ds.select(range(n_samples))
                ds = datasets.concatenate_datasets([ds, questions_to_keep])
        test_questions += ds['question']
        ds_all.append(ds)
    ds = datasets.concatenate_datasets(ds_all)
    # Add empty/none cot column
    ds = ds.map(lambda x: {"cot": None, **x})
    # Simple deduplication
    memory = set()
    def is_unique(elem, column, memory):
        if elem[column] in memory: return False
        memory.add(elem[column])
        return True
    # Drop duplicates in `ds` on "col1"
    # import pdb; pdb.set_trace()
    ds = ds.filter(partial(is_unique, column="question", memory=memory))
    ds.push_to_hub("simplescaling/s50K")
