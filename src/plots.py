import matplotlib.pyplot as plt


def fig_segment_distribution(segment_counts):
    fig, ax = plt.subplots()
    ax.bar(segment_counts.index.astype(str), segment_counts.values)
    ax.set_xlabel("Segment")
    ax.set_ylabel("Customers")
    ax.set_title("Customers per segment")
    return fig


def fig_top_categories(cat_counts, top_n=10):
    top = cat_counts.head(top_n)
    fig, ax = plt.subplots()
    ax.bar(top.index.astype(str), top.values)
    ax.set_xlabel("Category")
    ax.set_ylabel("Transactions")
    ax.set_title("Top categories")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def fig_monthly_spend(monthly_spend):
    fig, ax = plt.subplots()
    ax.plot(monthly_spend.index.astype(str), monthly_spend.values, marker="o")
    ax.set_xlabel("Month")
    ax.set_ylabel("Spend")
    ax.set_title("Monthly spend trend")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig
