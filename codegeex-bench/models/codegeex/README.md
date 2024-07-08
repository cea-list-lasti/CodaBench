# Setup

Run ```pip install -e .``` in codegeex folder


# Run inference

Run ```sbatch sbatch_inference.sh```
A HumanEval output is generated in output/

# Metrics

\( pass@k := E[1 − (n−c k ) / (n k ) ], n = 200, k ∈ {1, 10, 100} \)

where n is the total n umber of generation (n=200 in this work), k is the sampling budget (typically k ∈ {1, 10, 100}) and c is the number of samples that pass all test cases.

From the paper : 
> "We use temperature sampling (t ∈ [0, 1]) and nucleus sampling (p ∈ [0, 1])
For CodeGeeX in code generation, we use t = 0.2, p = 0.95 for pass@1 and t = 0.8, p = 0.95 for pass@10 and pass@100 (except for Go and JavaScript, where p = 0.9)."