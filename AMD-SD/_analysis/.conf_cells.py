MD = """
## 7. Is the model missing lesions, or finding them as the wrong class?

Pooled Dice can look good while small lesions are missed, because large lesions dominate it. This
section checks a second possibility: a lesion is **found, but labelled as a different fluid type**
(e.g. a small SRF lesion called PED, which sits right next to it).

For every expert lesion on the **original** validation scans:
- **found, right class**: >= 10% of its pixels predicted as the correct class (the detection used above)
- **found as any lesion**: >= 10% of its pixels predicted as *any* of SRF / IRF / PED / SHRM
- the difference = lesions found but with the wrong label
"""

CODE1 = """
rows = []
for j, i in enumerate(idx_val):
    gt, pr = masks[i], val_pred[j]
    any_lesion_pred = np.isin(pr, LESION_IDS)
    for c in LESIONS:
        k = CLASSES.index(c)
        comp, n = ndimage.label(gt == k, STRUCT)
        for q in range(1, n + 1):
            les = comp == q
            area = int(les.sum())
            if area < MIN_LESION_PX:
                continue
            labels_inside = pr[les]
            counts = np.bincount(labels_inside, minlength=len(CLASSES))
            rows.append({"class": c, "area_px": area,
                         "right_class": counts[k] / area >= 0.10,
                         "any_lesion": any_lesion_pred[les].mean() >= 0.10,
                         "mostly_predicted_as": CLASSES[int(counts.argmax())]})
conf = pd.DataFrame(rows)
conf["size_bin"] = pd.cut(conf["area_px"], BINS, labels=BIN_LABELS, right=False)

t = conf.groupby("size_bin", observed=False)[["right_class", "any_lesion"]].mean()
t["found but wrong class"] = t["any_lesion"] - t["right_class"]
t["n"] = conf.groupby("size_bin", observed=False).size()
print("Original validation scans, all classes")
t.round(3)
"""

CODE2 = """
t2 = conf.groupby("class")[["right_class", "any_lesion"]].mean()
t2["found but wrong class"] = t2["any_lesion"] - t2["right_class"]
print("By class (all sizes)")
print(t2.round(3).to_string())

missed = conf[~conf["right_class"]]
print(f"\\n{len(missed)} lesions not found as their own class. What did the model mostly call their pixels?")
print("(rows = true class, columns = most common predicted label inside the lesion, share of row)")
print(pd.crosstab(missed["class"], missed["mostly_predicted_as"], normalize="index").round(2).to_string())
"""

CODE3 = """
# Pixel-level confusion on original validation scans (ignore pixels excluded)
gt_all, pr_all = masks[idx_val].ravel(), val_pred.ravel()
keep = gt_all != IGNORE
cm = pd.crosstab(pd.Series(gt_all[keep], name="true").map(dict(enumerate(CLASSES))),
                 pd.Series(pr_all[keep], name="predicted").map(dict(enumerate(CLASSES))), normalize="index")
cm = cm.reindex(index=CLASSES, columns=CLASSES, fill_value=0)
fig, ax = plt.subplots(figsize=(6.5, 5))
im = ax.imshow(cm.values, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASSES, rotation=30, ha="right")
ax.set_yticks(range(len(CLASSES))); ax.set_yticklabels(CLASSES)
for r in range(len(CLASSES)):
    for c in range(len(CLASSES)):
        ax.text(c, r, f"{cm.values[r, c]:.2f}", ha="center", va="center", fontsize=8,
                color="white" if cm.values[r, c] > 0.5 else "black")
ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("Pixel confusion, original scans (row-normalised)")
plt.colorbar(im); plt.tight_layout(); plt.savefig(EVAL / "confusion_original.png", dpi=150); plt.show()
"""

MD2 = """
### How to read section 7

- If **found but wrong class** is large for small lesions, part of the "missed" small lesions are
  really label confusion: the model sees fluid but picks the wrong type.
- If it is near 0, the model genuinely does not see them: they are true misses.
- The crosstab shows where missed lesions go: mostly **background** = true misses; mostly another
  fluid class = confusion between neighbouring types (e.g. SRF vs PED).
"""
