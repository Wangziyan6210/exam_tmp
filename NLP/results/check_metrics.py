import json

state = json.load(open(r"C:\Users\nncc\exam_tmp\NLP\results\checkpoint-4689\trainer_state.json", "r"))
best = None
for entry in state.get("log_history", []):
    epoch = entry.get("epoch", "?")
    step = entry.get("step", "?")
    eval_acc = entry.get("eval_accuracy", None)
    eval_f1 = entry.get("eval_f1", None)
    eval_loss = entry.get("eval_loss", None)
    loss = entry.get("loss", None)
    if eval_acc is not None:
        print(f"Epoch {epoch} (step {step}): eval_accuracy={eval_acc:.4f}, eval_f1={eval_f1:.4f}, eval_loss={eval_loss:.4f}")
        if best is None or eval_acc > best[0]:
            best = (eval_acc, eval_f1, eval_loss)
    elif loss is not None:
        print(f"Epoch {epoch} (step {step}): train_loss={loss:.4f}")

if best:
    print(f"\nBest model: accuracy={best[0]:.4f}, f1={best[1]:.4f}, eval_loss={best[2]:.4f}")
