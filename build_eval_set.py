"""Write eval_set.jsonl and verify every evidence quote appears verbatim in its source article.

Each item: question, gold answer, source article, and `evidence` (verbatim text that answers it).
Evidence, not chunk ids, is the label so the same set scores every chunking strategy.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent

# (source, question, gold answer, evidence)
ITEMS = [
    ("AIXI", "Who first proposed AIXI, and in what year?",
     "Marcus Hutter, in 2000.",
     "AIXI was first proposed by Marcus Hutter in 2000"),
    ("AIXI", "Which two ideas does the AIXI formalism combine?",
     "Solomonoff induction and sequential decision theory.",
     "It combines Solomonoff induction with sequential decision theory."),
    ("AlphaChip", "What problem in chip design does AlphaChip automate?",
     "Macro placement, a step in chip floorplanning.",
     "AlphaChip is an automated method for macro placement, a crucial step in chip floorplanning."),
    ("AlphaChip", "Where was AlphaChip developed and what was it used to design?",
     "At Google Brain, where it was used in the design of Google's Tensor Processing Units.",
     "AlphaChip was developed at Google Brain, where it was used in the design of their Tensor Processing Units."),
    ("Curse_of_dimensionality", "Who coined the expression 'curse of dimensionality' and in what context?",
     "Richard E. Bellman, while considering problems in dynamic programming.",
     "The expression was coined by Richard E. Bellman when considering problems in dynamic programming."),
    ("Double_descent", "Who coined the term 'double descent' and when?",
     "Belkin et al., in 2019.",
     'The term "double descent" was coined by Belkin et al. in 2019'),
    ("Double_descent", "Where does the test error increase in double descent?",
     "Near the interpolation threshold, where the number of parameters equals the number of training data points.",
     "The increase usually occurs near the interpolation threshold, where the number of parameters is the same as the number of training data points"),
    ("ELMo", "What kind of neural network is ELMo and what inputs does it take?",
     "A bidirectional LSTM that takes character-level inputs and produces word-level embeddings.",
     "It is a bidirectional LSTM which takes character-level as inputs and produces word-level embeddings"),
    ("ELMo", "Which organizations created ELMo and when was it first released?",
     "The Allen Institute for Artificial Intelligence and the University of Washington, in February 2018.",
     "It was created by researchers at the Allen Institute for Artificial Intelligence, and University of Washington and first released in February 2018."),
    ("EfficientNet", "What is the key innovation of EfficientNet?",
     "Compound scaling, which uniformly scales depth, width and resolution with a single parameter.",
     "Its key innovation is compound scaling, which uniformly scales all dimensions of depth, width, and resolution using a single parameter."),
    ("Concept_drift", "What causes concept drift in a predictive model?",
     "The statistical properties of the target variable change over time in unforeseen ways.",
     "It happens when the statistical properties of the target variable, which the model is trying to predict, change over time in unforeseen ways."),
    ("Contrastive_Language_Image_Pre-training", "What does CLIP train, and with what kind of objective?",
     "A pair of neural networks, one for images and one for text, using a contrastive objective.",
     "CLIP) is a technique for training a pair of neural network models, one for image understanding and one for text understanding, using a contrastive objective."),
    ("Ball_tree", "What is a ball tree most notably used for?",
     "Nearest neighbor search.",
     "most notably nearest neighbor search."),
    ("Base_rate", "What method is used to integrate base rates with featural evidence?",
     "Bayes' rule.",
     "The method for integrating base rates and featural evidence is given by Bayes' rule."),
    ("Differentially_private_stochastic_gradient_descent", "Who introduced DP-SGD and at which conference?",
     "Abadi et al., at the 2016 ACM Conference on Computer and Communications Security.",
     "DP-SGD was introduced by Abadi et al. at the 2016 ACM Conference on Computer and Communications Security"),
    ("Differentially_private_stochastic_gradient_descent", "What does DP-SGD do to each per-example gradient before adding noise?",
     "It clips the L2 norm of each gradient to a threshold C.",
     "Then, it clips the L2 norm of each gradient to a threshold C."),
    ("Yoshua_Bengio", "Which award did Yoshua Bengio receive in 2018, and who shared it with him?",
     "The ACM A.M. Turing Award, shared with Geoffrey Hinton and Yann LeCun.",
     'Bengio received the 2018 ACM A.M. Turing Award, often referred to as the "Nobel Prize of Computing", with Geoffrey Hinton and Yann LeCun'),
    ("Croissant_metadata_format_", "Into which four layers does the Croissant metadata format divide metadata?",
     "Dataset Metadata, Resource, Structure and Semantic.",
     'divides metadata in four "layers": Dataset Metadata, Resource, Structure and Semantic'),
    ("Decision_tree_pruning", "How does pruning improve the predictive accuracy of a decision tree?",
     "It reduces the complexity of the final classifier, which reduces overfitting.",
     "Pruning reduces the complexity of the final classifier, and hence improves predictive accuracy by the reduction of overfitting."),
    ("80_Million_Tiny_Images", "How many images does the 80 Million Tiny Images dataset contain and at what size?",
     "79,302,017 color images of 32x32 pixels.",
     "It contains 79,302,017 32×32-pixel color images"),
    ("Bag-of-words_model", "What does the bag-of-words model disregard, and what does it keep?",
     "It disregards word order (and most syntax or grammar) but captures multiplicity.",
     "It disregards word order (and thus most of syntax or grammar) but captures multiplicity."),
    ("Confusion_matrix", "What do the rows and columns of a confusion matrix represent?",
     "Each row represents instances in an actual class and each column instances in a predicted class, or vice versa.",
     "Each row of the matrix represents the instances in an actual class while each column represents the instances in a predicted class, or vice versa"),
    ("Confusion_matrix", "Why is an actual-negative sample predicted as positive called a false positive?",
     "Because the negative sample is incorrectly identified by the classifier as being positive.",
     "This is called a false positive result because the negative sample is incorrectly identified by the classifier as being positive."),
    ("Concept_drift", "What simple method can detect data drift between replicas of a database?",
     "Running a checksum regularly.",
     "A simple drift detection is to run checksum regularly."),
    ("Decision_tree_pruning", "What is the horizon effect in decision tree growth?",
     "It is hard to tell when a tree algorithm should stop, because it is impossible to tell whether adding one more node will dramatically decrease error.",
     "However, it is hard to tell when a tree algorithm should stop because it is impossible to tell if the addition of a single extra node will dramatically decrease error. This problem is known as the horizon effect."),
]


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def main():
    bad = 0
    with open(ROOT / "eval_set.jsonl", "w", encoding="utf-8") as f:
        for i, (src, q, a, ev) in enumerate(ITEMS):
            text = norm((ROOT / "data" / f"{src}.txt").read_text(encoding="utf-8"))
            if norm(ev) not in text:
                print(f"EVIDENCE NOT FOUND [{i}] {src}: {ev[:70]}")
                bad += 1
            f.write(json.dumps({"id": i, "question": q, "answer": a, "source": f"{src}.txt",
                                "evidence": ev}, ensure_ascii=False) + "\n")
    print(f"{len(ITEMS)} items written, {bad} with unverifiable evidence")


if __name__ == "__main__":
    main()
