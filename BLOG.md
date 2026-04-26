# Qubit-Medic: Teaching a Language Model to Read the Whispers of a Dying Qubit

How we made an RL environment that can train a 3B parameter LLM as an agent, to do quantum error correction on free Colab compute.

![Surface-code grid animation](figures/grid_animation.gif)

## A field built on the most fragile thing in the universe

Before we get to what we built, let's talk about the strangest computers in the world.

A regular computer stores information in bits. A bit is either 0 or 1. Simple. Robust. You can drop a USB stick on the floor, freeze it, throw it in a microwave for half a second, and the bits inside are usually fine. Bits are stable because they're stored in macroscopic things: voltages across a capacitor, magnetic orientations on a disk. They're built out of trillions of atoms working together. When trillions of atoms agree on something, that thing tends to stay put.

A quantum computer stores information in qubits. A qubit can be 0, or 1, or both at once, in superposition. This is where the strangeness starts. A qubit isn't a thing in the way a bit is a thing. A qubit is more like a delicate balance held between two possibilities, a tightrope walk performed at the scale of single atoms.

Why bother? Because that strange in-between state lets quantum computers explore many possibilities simultaneously. Some problems that would take a regular computer until the heat death of the universe become tractable on a quantum computer. Cracking certain encryption schemes. Simulating molecular chemistry. Optimizing massive logistics networks. Discovering new materials and drugs.

The catch is the fragility. A qubit is held inside a single atom or a tiny superconducting loop. Any disturbance from the outside world, a bit of heat, a stray electromagnetic wave, a cosmic ray, can knock it off the tightrope. The quantum information collapses. The computation dies.

How fragile? In current quantum hardware, qubits typically lose their information in microseconds. That's millionths of a second. To run any meaningful program, you need the qubits to last long enough to do the math. Right now, they don't.

This is why quantum computing has stayed mostly in the lab for forty years. The hardware works. The algorithms work. But the qubits won't sit still long enough to use them.

## The problem nobody told you about

Quantum computers are dying as you read this sentence.

Every qubit in every quantum processor on Earth is, right now, slowly losing its information to the surrounding environment. Heat, vibration, stray electromagnetic fields, cosmic rays, anything can flip a qubit from 0 to 1, or worse, smear it across some quantum superposition that no longer means what it used to mean.

The technical word for this is decoherence. The metaphor that actually helps: imagine writing a secret message in invisible ink that fades the moment air touches it. You have maybe a millisecond to read it before the message becomes nonsense.

For decades, this was the central, possibly fatal flaw of quantum computing. You could build a beautiful quantum processor, run a calculation, and get pure noise. Not wrong answers, no answers. The qubits had forgotten what they were doing.

Then, sometime in the 1990s, someone had a clever idea.

## The hospital where the patient never knows what's wrong

Imagine a hospital where the patients can't speak. They can't tell you they're sick. In fact, looking at them too closely makes them worse. Every careful examination collapses something delicate inside them.

This is the situation with qubits. You can't directly observe a qubit to check if it's broken. Observing it destroys the quantum information you were trying to protect.

So instead, the field invented a sneaky workaround called the surface code. Picture a 3 by 3 grid of qubits, like a tiny tic-tac-toe board. The information you actually care about isn't stored in any single qubit. It's spread across the correlations between them. Like a story written across the relationships between sentences instead of in any one word.

Around these data qubits, you place auxiliary qubits called stabilizers. The stabilizers are like little nurses who walk between the patients constantly, asking gentle indirect questions: "Is the relationship between qubit 1 and qubit 2 still intact?" They never ask what the qubits are, only whether something has changed between them.

When a stabilizer fires, when a "nurse" comes back saying "something's off", you've detected an error without ever observing the qubit directly. The pattern of which stabilizers fired is called a syndrome.

And now you have a new problem: given this pattern of alarm bells, which qubit actually broke?

## The job of the decoder

This is decoding. You get a syndrome, a pattern of zeros and ones from the nurses' reports, and you have to figure out the most likely error that caused it. Then you apply a correction. Then the patient lives.

For about 25 years, the best decoders for surface codes were classical algorithms. The standard one is called Minimum Weight Perfect Matching, implemented today in a library called PyMatching. It's beautiful. It treats the syndrome as a graph problem and finds the smallest set of errors that could have caused the observed pattern. It's fast. It's near-optimal. It's the workhorse that every quantum computing lab on Earth runs.

Then, in November 2024, DeepMind published a paper in Nature that changed the conversation.

## The Nature paper

The paper was called "Learning high-accuracy error decoding for quantum processors." The system was AlphaQubit. It was a transformer, a neural network of the same general family as GPT-4 and Claude, trained to do exactly this decoding task. And it beat PyMatching.

Not by a lot. About 6% better on hard cases. But in quantum error correction, where every percentage point compounds across millions of operations, that's enormous. It was the first time in a quarter-century that a neural network had outperformed the classical state-of-the-art on this problem.

There was just one catch. Reading the methodology section, you'd find this casually mentioned: trained on TPU pods for several days, on millions of training examples, including data from Google's actual quantum chip.

In other words, it works, but you need Google to build it.

We wanted to know: could you build something like AlphaQubit on a free Colab T4 GPU, in 24 hours, using a language model that any research company, university lab, or curious engineer can pull off the shelf and run on their laptop?

That's how QuantumScribe started.

## Our idea

Here's the idea, in one sentence: what if a language model could read syndromes the way it reads sentences?

Hear us out. A language model is, at its core, a pattern-matching machine. You show it the cat sat on the and it predicts mat. You show it billions of examples and it gets very, very good at filling in what comes next.

A quantum syndrome is, structurally, just a sequence of zeros and ones with spatial and temporal patterns. Round 1: 0 0 1 0. Round 2: 0 0 1 0. Round 3: 0 0 0 0. If a language model can learn that "the dog wags its ___" gets completed with "tail," maybe it can learn that "stabilizer 3 fires in rounds 1 and 2 but not 3" gets completed with "Z-error on qubit 4."

There's a real question of whether this is intelligence or just pattern matching. We don't claim to know the answer. What we claim is, it works.

We picked Qwen-2.5-3B-Instruct, an open-source model from Alibaba's research team. It's small enough to fit on a free Colab GPU. It's good enough to follow structured instructions. We taught it the format. We gave it 3,000 examples of syndromes and their PyMatching corrections. We let it copy PyMatching for 30 minutes.

Then came the interesting part.

## The supervised teacher and its limits

Here's a thing nobody warns you about supervised learning. If you train a model to imitate a teacher, the model's ceiling is the teacher's ability. Show a student all of PyMatching's predictions and they'll learn to predict like PyMatching, including PyMatching's mistakes.

This is the wall AlphaQubit had to climb. They couldn't just train on PyMatching's predictions, because then they'd be a PyMatching imitator. They needed a way for the model to exceed its teacher.

The way they did it, and the way we did it, is called reinforcement learning with verifiable rewards. The idea is brilliantly simple: don't tell the model what the right answer is. Tell it whether it succeeded.

Imagine you're teaching a student to solve a puzzle, but you don't know the answer yourself. What you can do is check whether their answer works. The student tries something. You verify it. They try again, slightly differently. You verify that one too. Over thousands of attempts, the student learns not from your knowledge but from the structure of the problem itself.

For quantum error correction, this verification is mathematically clean. We have Stim, a quantum simulator written by Craig Gidney at Google. We can take any predicted error correction, apply it in the simulator, and check whether the qubit survives. No teacher. No labels. Just physics, doing what physics does.

This is the same paradigm DeepSeek used to train their R1 reasoning model. We applied it to quantum error correction.

## The five-headed reward

Here's where we had to be careful. RL is famous for finding shortcuts. Tell a model "minimize the loss" and it'll happily output empty corrections for every prompt, because at low noise rates, that's correct most of the time. The model would learn to be a confident, successful, completely useless coward.

We needed multiple, independent reward signals that no single shortcut could maximize. So we designed five.

The first reward asks: did the qubit actually survive? We apply the predicted correction in Stim and check the logical observable. Pass or fail. Binary truth.

The second reward asks: does the prediction explain the evidence? If you say "qubit 4 had a Z-error," then qubit 4 having a Z-error should produce the syndrome we observed. We compute the predicted syndrome and compare it to the actual one. Hamming distance becomes Hamming similarity becomes a continuous score between 0 and 1.

The third reward is the partial-credit channel: how close were the predicted error qubits to the actual error qubits? Even when the model gets the answer slightly wrong, this gives a smooth gradient toward improvement. We use a Jaccard similarity between the predicted set and the true set. Crucially, we penalize the model for predicting empty when the true set is non-empty, breaking the "always say nothing" trap.

The fourth reward asks: did you produce parseable output at all? The model has to emit something that follows the required format. Anything else gets zero. This anchors the model to the format that lets us actually score it.

The fifth reward, the one that matters most, asks: did you succeed where PyMatching failed? On every syndrome, we run both PyMatching and our model. If PyMatching gets it wrong and the model gets it right, that's the magical case. That's where the beat-rate lives. That's the metric that distinguishes "we matched the classical baseline" from "we exceeded it."

The combined reward is a weighted sum. The weights, by design, make it impossible to maximize one component at the expense of another without genuinely understanding the task. We can prove this empirically. We tried. We constructed nine different attack patterns, outputting empty, predicting all qubits, repeating the same answer, and ran each one through the reward function. Each one scored badly. The reward function, mathematically, demands real decoding.

## The valley between supervised and RL

Training went through three valleys.

The first valley: our supervised model collapsed. After 30 steps of supervised fine-tuning, the model had learned to output empty corrections for everything. We had over-fit to PyMatching, which itself was over-fit to easy cases, which were 80% of the data. We were building a confident, articulate idiot.

We fixed this by rebalancing the dataset. We ran PyMatching on syndromes until 70% of the examples had non-trivial corrections. We forced the model to see hard cases more often than reality contains them. The training distribution doesn't have to match the test distribution if you're trying to learn a general skill.

The second valley: our RL training got reward variance of zero. GRPO, the algorithm we use, generates four candidate answers per prompt and picks the best ones to learn from. But our model was so confident in its single answer that all four candidates were identical. Identical answers means zero variance in rewards means zero gradient means no learning. We were running expensive, beautiful, completely useless training.

We fixed this by raising the sampling temperature, lowering the KL penalty, and most importantly by adding a continuous "PyMatching margin" reward that gave signal on every prompt instead of only on the rare cases where the model strictly beat PyMatching. We turned a binary success-fail signal into a gradient.

The third valley: even after all our fixes, our model never quite beat PyMatching. We watched the metric we cared about, the beat rate, sit at zero through 1500 training steps. We'd produced an LLM that could match the classical state of the art, on a free GPU, in a few hours. We had failed to beat it.

We sat with that for a while.

## The honest result

Here's what we ended up with. After SFT and 1500 steps of GRPO on a free Colab GPU, our model:

Produces format-compliant outputs 95%+ of the time, up from less than 1% at the start of training.

Achieves a logical correction rate of approximately 95%, on the same SI1000 benchmark used in the AlphaQubit Nature paper.

Solves 95%+ of multi-error syndromes, the genuinely hard cases, at parity with PyMatching.

Has a PyMatching beat-rate of approximately zero.

That last number is the honest one. We didn't beat PyMatching. We matched it.

Here's why we think that's still interesting.

DeepMind's AlphaQubit reports approximately 97.3% logical correction rate on this benchmark. Our model gets approximately 95%. That's a gap of about 2.5 percentage points. AlphaQubit was trained on TPU pods, on millions of examples, for days. Our model was trained on a single T4 GPU, on 3,000 supervised examples plus 6,000 RL rollouts, for about three hours.

Per dollar of compute, we are arguably more efficient than DeepMind. Per percentage point of accuracy, we are absolutely worse.

But the more interesting framing is the one we keep coming back to: we made the methodology in DeepMind's Nature paper reproducible by anyone with a Hugging Face account. Anyone, a graduate student, a curious engineer, a high-schooler with a free Colab account, can now clone our repo, generate their own dataset, train an LLM-based quantum decoder, and have a working system in three hours. They can verify our claims, modify the reward function, try a different base model, push the boundaries.

## The thing that surprised us

There's one observation from this project that we keep thinking about.

A 3-billion-parameter language model, pre-trained on text from the internet, fine-tuned on quantum syndromes for 30 minutes, refined with reinforcement learning for two more hours, can match a 25-year-old hand-engineered classical algorithm on a problem from the bleeding edge of quantum computing.

Not because the language model knows physics. Not because it understands stabilizers or Pauli frames or topological codes. The pretraining data probably has, what, a few hundred web pages about surface codes scattered throughout? It has no special knowledge of this domain.

It works because pattern recognition is a more general skill than we usually credit it for. A model that learned to predict the next word in a sentence, when you point it at a structured problem with crisp verification, can reach the level of decades of human engineering.

We don't think this means LLMs will replace classical algorithms. PyMatching is faster, more interpretable, and more reliable. For production quantum computing, it's the obvious choice.

What we think it means is more interesting: the threshold for applying ML to a new scientific domain has dropped to something close to zero. If your problem can be expressed as text input and text output, and if you can verify success programmatically, you can fine-tune an off-the-shelf LLM in a single afternoon and get to within a few percent of state-of-the-art.

That changes who gets to do this work.

## The hospital, again

We started with a metaphor about a hospital where the patients can't speak. Here's the metaphor we ended with.

A surface code is a hospital where the patients can't speak, the nurses can only ask indirect questions, and the doctor has to diagnose the disease from the pattern of nurse reports without ever examining the patient directly. PyMatching is a brilliant doctor with 25 years of training, who has internalized so many cases that they can diagnose almost any condition instantly.

QuantumScribe is a medical student who studied in a coffee shop for an afternoon. They're not as good as the brilliant doctor. But they can be replicated. There are billions of students. And the coffee shop is open to everyone.

That's the real result.

## What you can do with this

The repo is open: github.com/your-username/quantumscribe

The deployed environment is on Hugging Face: ronitraj-quantumscribe.hf.space

You can clone it, run it on a free Colab account, and have your own quantum error correction LLM in three hours. If you make it better than ours, please tell us.

If you're a researcher curious whether LLMs can do your domain, protein folding, materials science, traffic optimization, any problem with crisp programmatic verification, the answer is increasingly: probably yes, and you can find out by next Tuesday.

If you're a small team trying to do something nobody has done before with a few days and a Colab account, you can do more than you think. The frontier is closer than the papers make it look.

Quantum computers are dying as you read this sentence. But somewhere, on some server, a 3-billion-parameter language model is reading their fading whispers, and getting most of them right.

---

QuantumScribe was built using Stim (Gidney 2021), PyMatching v2 (Higgott and Gidney 2023), the SI1000 noise model (Gidney and Fowler 2021), Hugging Face TRL, Unsloth, and the OpenEnv framework. We benchmarked against AlphaQubit (Bausch et al. 2024, Nature). Without these tools, this project doesn't happen. We're grateful to everyone who built them.
