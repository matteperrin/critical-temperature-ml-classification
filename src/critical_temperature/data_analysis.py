"""Full-dataset exploratory plots and tables; not model feature selection."""
from pathlib import Path

import matplotlib.pyplot as plt

if __package__:
    from .phase_one_reports import PROJECT_ROOT, generate_reports, load_inputs
else:
    from phase_one_reports import PROJECT_ROOT, generate_reports, load_inputs


def run_analysis(root=PROJECT_ROOT, *, train=None, unique=None, processed=None):
    """Regenerate Phase I outputs without modifying raw or processed inputs."""
    root = Path(root)
    train, unique, df = load_inputs(root, train=train, unique=unique, processed=processed)
    tables = generate_reports(root, train=train, unique=unique, processed=df)
    figures_dir = root / "reports/figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    def save(fig, filename):
        fig.tight_layout()
        fig.savefig(figures_dir / filename, dpi=300)
        plt.close(fig)

    fig, ax = plt.subplots()
    balance = tables["class_balance.csv"]
    ax.bar(balance.above_77k, balance.record_count)
    ax.set(title="Class Distribution", xlabel="Above 77 K", ylabel="Number of Records")
    ax.set_xticks([0, 1], ["0 = 77 K or below", "1 = Above 77 K"])
    save(fig, "class_distribution.png")

    fig, ax = plt.subplots()
    ax.hist(df.critical_temp, bins=30)
    ax.set(title="Distribution of Critical Temperature", xlabel="Critical Temperature (K)", ylabel="Frequency")
    save(fig, "critical_temperature_histogram.png")

    fig, ax = plt.subplots()
    ax.boxplot(df.critical_temp.dropna())
    ax.set(title="Critical Temperature Boxplot", ylabel="Critical Temperature (K)")
    save(fig, "critical_temperature_boxplot.png")

    # Correlation ranking is used only for this exploratory visualization.
    columns = tables["feature_target_correlations.csv"].dropna(subset=["pearson_correlation"]).feature.head(10).tolist()
    columns += ["critical_temp"]
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(df[columns].corr(), aspect="auto", vmin=-1, vmax=1, cmap="coolwarm")
    fig.colorbar(image, ax=ax, label="Pearson Correlation")
    ax.set_xticks(range(len(columns)), columns, rotation=90)
    ax.set_yticks(range(len(columns)), columns)
    ax.set_title("Exploratory Correlation Heatmap")
    save(fig, "correlation_heatmap.png")

    fig, ax = plt.subplots()
    ax.scatter(df.wtd_mean_Valence, df.critical_temp, alpha=.4)
    ax.set(title="Weighted Mean Valence vs Critical Temperature", xlabel="Weighted Mean Valence", ylabel="Critical Temperature (K)")
    save(fig, "weighted_mean_valence_scatter.png")

    selected = [name for name in ("wtd_mean_Valence", "wtd_std_ThermalConductivity", "number_of_elements") if name in df]
    fig, axes = plt.subplots(1, len(selected), figsize=(5 * len(selected), 4), squeeze=False)
    for ax, feature in zip(axes.flat, selected):
        for label in (0, 1):
            values = df.loc[df.above_77k.eq(label), feature].dropna()
            if not values.empty:
                ax.hist(values, bins=25, alpha=.5, density=True, label=f"above_77k = {label}")
        ax.set(xlabel=feature, ylabel="Density")
        ax.legend()
    fig.suptitle("Exploratory Class-conditional Feature Distributions")
    save(fig, "class_feature_distribution.png")

    elements = unique.drop(columns=["critical_temp", "material"]).select_dtypes(include="number")
    complexity = elements.gt(0).sum(axis=1)
    fig, ax = plt.subplots()
    for label in (0, 1):
        mask = df.above_77k.eq(label)
        ax.scatter(complexity[mask], train.loc[mask, "critical_temp"], alpha=.25, label=f"above_77k = {label}")
    ax.set(title="Composition Complexity vs Critical Temperature", xlabel="Number of Present Elements", ylabel="Critical Temperature (K)")
    ax.legend()
    save(fig, "composition_complexity_temperature.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    prevalence = tables["element_prevalence.csv"].head(20)
    ax.bar(prevalence.element, prevalence.record_proportion)
    ax.set(title="Most Common Elements", xlabel="Element", ylabel="Proportion of Records Containing Element")
    save(fig, "element_prevalence.png")
    return tables


def main():
    tables = run_analysis()
    for filename, table in tables.items():
        print(f"\n--- {filename} ---")
        print(table.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
