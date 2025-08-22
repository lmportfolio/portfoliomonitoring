
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter, StrMethodFormatter
import numpy as np
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
import re



st.title("Portfolio Monitoring Analytics")

st.set_page_config(
    page_title="Portfolio Monitoring Analytics",
    layout="centered"
)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Data Upload",
                                        "Metric Performance Visuals",
                                        "Percentage Performance Visuals",
                                        "Metric Performance Clusters",
                                        "Percentage Performance Clusters",
                                        "Sentiment Analysis",])

with tab1:
    st.header("1) Upload your portfolio file")
    st.text("By uploading any Input Sheet or Portfolio Monitoring, a script will clean any extra columns, and 'clean' the dataset. Please ensure that upload was successful by revising the table below. All empty values under outcome or execution status will be considered to be gray. Any old xLOB will be renamed for the analysis purpose (ex. Medical to MSP).")
    st.text("The 'Stage Level' variable will be created, which is defined as the number of months a project has been active. Additionally, year-to-date difference, or YTD Diff, will be used for metric performance. This takes in the planned target and subtracts it from the actual savings of a project. A positive value indicates a project exceeding its goal, a negative value indicates an underperformance.")

    uploaded = st.file_uploader(
        "Choose a CSV or Excel file",
        type=["csv", "xls", "xlsx"]
    )
    if not uploaded:
        st.info("Awaiting file upload…")
    else:
        try:
            if uploaded.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded, parse_dates=["Date"])
            else:
                df = pd.read_excel(uploaded,
                                   parse_dates=["Date"],
                                   engine="openpyxl")

            df.rename(columns={
                "YTD Plan (do not change)":        "YTD Plan",
                "If outcome is Y/R, why?":         "Outcome Comments",
                "Outcome":                          "Outcome Status",
                "If execution status is Y/R - Why?": "Execution Comments",
                "Outcome Status Details/Comments": "Outcome Comments",
                "Execution Status Description" : "Execution Comments"
            }, inplace=True)

            keep_cols = [
                "Date","LOB","Loss/Expense","Initiative/Epic",
                "Execution Status","Execution Comments",
                "Outcome Status","Outcome Comments",
                "Path to Green","YTD Savings","YTD Plan"
            ]
            missing = set(keep_cols) - set(df.columns)
            if missing:
                st.error(f"Missing required columns: {missing}")
                st.stop()
            port_mon = df[keep_cols].copy()


            port_mon["Date"] = pd.to_datetime(
                port_mon["Date"], errors="coerce"
            ).dt.to_period("M").dt.to_timestamp()
            port_mon["Date"] = port_mon["Date"].dt.normalize()
            df["Year"] = df["Date"].dt.year.astype(str)
            df["Month"] = df["Date"].dt.month_name().str[:3]


            valid = {"Red","Yellow","Gray","Green"}
            port_mon["Outcome Status"]   = port_mon["Outcome Status"]  \
                .where(port_mon["Outcome Status"].isin(valid), "Gray")
            port_mon["Execution Status"] = port_mon["Execution Status"] \
                .where(port_mon["Execution Status"].isin(valid), "Gray")


            port_mon["LOB"] = port_mon["LOB"].replace({
                "Medical":"MSP","SI":"MSP","Centralized":"xLOB"
            })


            codes = pd.factorize(port_mon["Initiative/Epic"])[0] + 1
            port_mon["Initiative/Epic ID"] = (
                pd.Series(codes, index=port_mon.index)
                  .astype(str)
                  .str.zfill(5)
            )


            port_mon = port_mon.sort_values(
                ["Initiative/Epic ID", "Date"]
            ).reset_index(drop=True)

            port_mon["Stage Level"] = (
                port_mon
                .groupby("Initiative/Epic ID")
                .cumcount()
                .add(1)
            )

            st.session_state["port_mon"] = port_mon
            st.success("✅ Portfolio Monitoring DataFrame is ready!")
            disp = port_mon.copy()
            disp["Date"] = disp["Date"].dt.strftime("%Y-%m-%d")

            st.dataframe(disp, use_container_width=True)

        except Exception as e:
            st.error(f"Something went wrong: {e}")


with tab2:
    st.header("2) YTD Difference Performance")
    st.text(
        "The purpose of this analysis is to use the available metrics "
        "to understand the different projects’ performance throughout the months. "
        "We exclude Service and xLOB because of the missing financial data percentage."
    )
    st.markdown("""
    **Objective:**  
    • Exclude projects where **either** Execution Status or Outcome Status = “Gray”   
    • Exclude LOBs xLOB and Service   
    • Show boxplots by Fiscal Year and by Month    
    • User can filter out LOB when needed    
    • User is able to visualize multiple projects  
    
    """)

    port_mon = st.session_state.get("port_mon")
    if port_mon is None:
        st.warning("Please go to Tab 1 and upload/clean your data first.")
        st.stop()
    df = port_mon.copy()

    df = df.loc[
        (df["Outcome Status"]   != "Gray") &
        (df["Execution Status"] != "Gray") &
        (~df["LOB"].isin(["xLOB","Service"]))
    ].reset_index(drop=True)

    df["Year"]     = df["Date"].dt.year.astype(str)
    df["Month"]    = df["Date"].dt.month_name().str[:3]
    df["YTD_diff"] = df["YTD Savings"] - df["YTD Plan"]

    df.dropna(subset=["YTD_diff"], inplace=True)
    if df.empty:
        st.warning("No data left after dropping rows with missing financials.")
        st.stop()

    years    = ["All"] + sorted(df["Year"].unique())
    year_sel = st.selectbox("Select Year (or All):", years)
    show_y   = st.checkbox("Show Yearly boxplot",   True)
    show_m   = st.checkbox("Show Monthly boxplots", True)

    df_year = df if year_sel=="All" else df[df["Year"]==year_sel]
    if df_year.empty:
        st.warning(f"No data available for {year_sel}.")
        st.stop()

    if show_y:
        fig, ax = plt.subplots(figsize=(6,4))
        sns.boxplot(
            x="Year", y="YTD_diff", data=df_year,
            palette="Set2", showfliers=False, ax=ax
        )
        ax.set_title("YTD Savings – YTD Plan by Fiscal Year")
        ax.set_xlabel("Year")
        ax.set_ylabel("Difference ($)")
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda v, _: f"${v:,.0f}")
        )
        st.pyplot(fig)

    if show_m:
        if year_sel == "All":
            g = sns.catplot(
                x="Month", y="YTD_diff", col="Year",
                data=df_year, kind="box", showfliers=False,
                col_wrap=3, height=3.5, aspect=1, palette="Set3"
            )
            for ax in g.axes.flatten():
                ax.tick_params(axis="x", rotation=30)
                ax.yaxis.set_major_formatter(
                    FuncFormatter(lambda v, _: f"${v:,.0f}")
                )
            g.fig.subplots_adjust(top=0.85)
            g.fig.suptitle("YTD Savings – YTD Plan by Month & Year")
            st.pyplot(g.fig)
        else:
            fig, ax = plt.subplots(figsize=(7,4))
            sns.boxplot(
                x="Month", y="YTD_diff", data=df_year,
                palette="Set3", showfliers=False, ax=ax
            )
            ax.set_title(f"YTD Savings – YTD Plan by Month ({year_sel})")
            ax.set_xlabel("Month")
            ax.set_ylabel("Difference ($)")
            ax.tick_params(axis="x", rotation=30)
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda v, _: f"${v:,.0f}")
            )
            st.pyplot(fig)

    st.subheader("Average YTD Difference by LOB")
    lob_options = sorted(df_year["LOB"].unique())
    lob_sel     = st.multiselect(
        "Select LOB(s):", lob_options, default=lob_options
    )
    df_lob = df_year[df_year["LOB"].isin(lob_sel)]

    if not df_lob.empty:
        lob_mean = df_lob.groupby("LOB")["YTD_diff"]\
                         .mean().reindex(lob_options)
        colors   = ["green" if v>=0 else "red" for v in lob_mean]

        fig, ax = plt.subplots(figsize=(6, max(4,len(lob_mean)*0.5)))
        sns.barplot(
            x=lob_mean.values, y=lob_mean.index,
            palette=colors, edgecolor="black", ax=ax
        )
        ax.axvline(0, color="black", linewidth=1)
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda v, _: f"${v:,.0f}")
        )
        ax.set(title="Avg YTD Savings – YTD Plan by LOB",
               xlabel="", ylabel="LOB")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        for sp in ["top","right"]:
            ax.spines[sp].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)

        st.subheader("Heatmap: Avg YTD Difference by Month & LOB")
        heat = (
            df_lob.groupby(["LOB","Month"])["YTD_diff"]
                  .mean()
                  .unstack("Month")
        )
        heat = heat.loc[:, heat.notna().any(axis=0)]

        mon_order = ["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"]
        heat = heat.reindex(columns=[m for m in mon_order if m in heat.columns])

        fig, ax = plt.subplots(
            figsize=(8, max(3, heat.shape[0] * 0.5))
        )
        sns.heatmap(
            heat,
            annot=False,
            cmap="RdYlGn_r",  # use built‐in red→white→green
            center=0,  # force the midpoint at zero
            cbar_kws={
                "label": "Avg YTD_diff ($)",
                "format": StrMethodFormatter("${x:,.0f}")
            },
            ax=ax
        )
        st.pyplot(fig)

    st.subheader("Project‐level YTD Difference Over Time")
    projs = sorted(df_year["Initiative/Epic"].unique())
    sel   = st.multiselect("Highlight project(s):", projs)

    fig_ts = px.line(
        df_year, x="Date", y="YTD_diff",
        line_group="Initiative/Epic",
        color_discrete_sequence=["lightgray"]
    )
    if sel:
        df_sel  = df_year[df_year["Initiative/Epic"].isin(sel)]
        fig_sel = px.line(
            df_sel, x="Date", y="YTD_diff",
            color="Initiative/Epic", markers=True
        )
        for trace in fig_sel.data:
            fig_ts.add_trace(trace)
        fig_ts.update_layout(showlegend=True)

    fig_ts.update_layout(
        title="YTD Savings – YTD Plan Over Time",
        xaxis_title="Date",
        yaxis_title="Difference ($)",
        showlegend=bool(sel)
    )
    st.plotly_chart(fig_ts, use_container_width=True)


with tab3:
    st.header("3) YTD % Difference Performance")
    st.text(
        "Exact same filters & charts as Tab 2, but plotting the true % change "
        "between consecutive periods (dropping each project’s first period)."
    )
    st.markdown("""
    **Objective:**  
    • Exclude Execution or Outcome = Gray  
    • Exclude LOBs xLOB, Service  
    • Compute YTD_diff = YTD Savings – YTD Plan  
    • Then YTD_diff_pct = 100 * (Current YTD_Diff – Previous YTD_Diff)/Previous YTD_Diff  
    • Drop the first record per project  
    • Show yearly & monthly boxplots, bar chart, heatmap & time series
    """)


    pm = st.session_state.get("port_mon")
    if pm is None:
        st.warning("Please complete Tab 1 first.")
        st.stop()
    df3 = pm.copy()

    df3 = df3.loc[
        (df3["Outcome Status"] != "Gray") &
        (df3["Execution Status"] != "Gray") &
        (~df3["LOB"].isin(["xLOB", "Service"]))
        ].reset_index(drop=True)

    df3["Year"] = df3["Date"].dt.year.astype(str)
    df3["Month"] = df3["Date"].dt.month_name().str[:3]
    df3["YTD_diff"] = df3["YTD Savings"] - df3["YTD Plan"]

    df3 = df3.sort_values(["Initiative/Epic", "Date"])
    df3["prev_diff"] = df3.groupby("Initiative/Epic")["YTD_diff"].shift(1)
    df3["Pct_Improve"] = (df3["YTD_diff"] - df3["prev_diff"]) \
                         / df3["prev_diff"].abs() * 100

    df3 = df3.dropna(subset=["prev_diff", "Pct_Improve"])
    df3 = df3.replace([np.inf, -np.inf], np.nan).dropna(subset=["Pct_Improve"])
    if df3.empty:
        st.warning("No data after dropping first‐period rows.")
        st.stop()

    threshold = st.slider(
        "Outlier cutoff for % improvement (|Pct_Improve| > X%)",
        min_value=0, max_value=2000, value=500, step=50, key="tab3_outlier_pct"
    )
    outliers = df3.loc[df3["Pct_Improve"].abs() > threshold]
    df_main = df3.loc[df3["Pct_Improve"].abs() <= threshold]
    st.write(f"Excluded {len(outliers)} row(s) with |Pct_Improve| > {threshold}%.")

    years3 = ["All"] + sorted(df_main["Year"].unique())
    year_sel3 = st.selectbox("Select Year (or All):", years3, key="tab3_year")
    show_y3 = st.checkbox("Show yearly boxplot", True, key="tab3_show_y")
    show_m3 = st.checkbox("Show monthly boxplots", True, key="tab3_show_m")

    df3_y = df_main if year_sel3 == "All" else df_main[df_main["Year"] == year_sel3]
    if df3_y.empty:
        st.warning(f"No data for {year_sel3}.")
        st.stop()

    if show_y3:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(
            x="Year", y="Pct_Improve", data=df3_y,
            palette="Set2", showfliers=False, ax=ax
        )
        ax.set_title("YTD % Improvement by Fiscal Year")
        ax.set_xlabel("Year")
        ax.set_ylabel("Pct Improvement")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}%"))
        st.pyplot(fig)

    if show_m3:
        if year_sel3 == "All":
            g = sns.catplot(
                x="Month", y="Pct_Improve", col="Year",
                data=df3_y, kind="box", showfliers=False,
                col_wrap=3, height=3.5, aspect=1, palette="Set3"
            )
            for ax in g.axes.flatten():
                ax.tick_params(axis="x", rotation=30)
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}%"))
            g.fig.subplots_adjust(top=0.85)
            g.fig.suptitle("YTD % Improvement by Month & Year")
            st.pyplot(g.fig)
        else:
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.boxplot(
                x="Month", y="Pct_Improve", data=df3_y,
                palette="Set3", showfliers=False, ax=ax
            )
            ax.set_title(f"YTD % Improvement by Month ({year_sel3})")
            ax.set_xlabel("Month")
            ax.set_ylabel("Pct Improvement")
            ax.tick_params(axis="x", rotation=30)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}%"))
            st.pyplot(fig)

    st.subheader("Average % Improvement by LOB")
    lob_opts3 = sorted(df3_y["LOB"].unique())
    lob_sel3 = st.multiselect(
        "Select LOB(s):", lob_opts3, default=lob_opts3, key="tab3_lob"
    )
    df3_lob = df3_y[df3_y["LOB"].isin(lob_sel3)]

    if not df3_lob.empty:
        lob_mean3 = df3_lob.groupby("LOB")["Pct_Improve"] \
            .mean().reindex(lob_opts3)
        cols3 = ["green" if v >= 0 else "red" for v in lob_mean3]

        fig, ax = plt.subplots(figsize=(6, max(4, len(lob_mean3) * 0.5)))
        sns.barplot(
            x=lob_mean3.values, y=lob_mean3.index,
            palette=cols3, edgecolor="black", ax=ax
        )
        ax.axvline(0, color="black", linewidth=1)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}%"))
        ax.set(title="Avg YTD % Improvement by LOB", xlabel="", ylabel="LOB")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)

        st.subheader("Heatmap: Avg % Improvement by Month & LOB")
        heat3 = (
            df3_lob.groupby(["LOB", "Month"])["Pct_Improve"]
            .mean().unstack("Month")
        )
        heat3 = heat3.loc[:, heat3.notna().any(axis=0)]
        mon_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        heat3 = heat3.reindex(columns=[m for m in mon_order if m in heat3.columns])

        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            "rwg", ["red", "white", "green"], N=256
        )
        vmin, vmax = heat3.min().min(), heat3.max().max()
        vcenter = 0 if vmin <= 0 <= vmax else (vmin + vmax) / 2
        norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

        fig, ax = plt.subplots(
            figsize=(8, max(3, heat3.shape[0] * 0.5))
        )
        sns.heatmap(
            heat3, annot=False, cmap=cmap, norm=norm,
            cbar_kws={"label": "Avg % Improvement",
                      "format": StrMethodFormatter("{x:.1f}%")},
            ax=ax
        )
        ax.set(xlabel="Month", ylabel="LOB")
        st.pyplot(fig)

    st.subheader("Project‐level % Improvement Over Time")
    projs3 = sorted(df3_y["Initiative/Epic"].unique())
    sel3 = st.multiselect("Highlight project(s):", projs3, key="tab3_proj")

    fig_ts3 = px.line(
        df3_y, x="Date", y="Pct_Improve",
        line_group="Initiative/Epic",
        color_discrete_sequence=["lightgray"]
    )
    if sel3:
        df_sel3 = df3_y[df3_y["Initiative/Epic"].isin(sel3)]
        fig_sel3 = px.line(
            df_sel3, x="Date", y="Pct_Improve",
            color="Initiative/Epic", markers=True
        )
        for tr in fig_sel3.data:
            fig_ts3.add_trace(tr)
        fig_ts3.update_layout(showlegend=True)

    fig_ts3.update_layout(
        title="YTD % Improvement Over Time",
        xaxis_title="Date",
        yaxis_title="Pct Improvement (%)",
        yaxis_tickformat=".1f%",
        showlegend=bool(sel3)
    )
    st.plotly_chart(fig_ts3, use_container_width=True)

    if not outliers.empty:
        st.subheader("Rows Excluded as Extreme Outliers")

        st.markdown(
            """
            **Field definitions:**  
            • **Previous Month Diff**: prior period’s YTD difference (YTD Savings – YTD Plan).  
            • **Pct Improvement**:  
              100 × (Current YTD Diff – Previous Month Diff) / |Previous Month Diff|  
            """
        )

        display = outliers[
            ["Initiative/Epic", "Date", "LOB", "prev_diff", "YTD_diff", "Pct_Improve"]
        ].copy()
        display.rename(columns={
            "prev_diff": "Previous Month Diff",
            "YTD_diff": "Current YTD Diff",
            "Pct_Improve": "Pct Improvement"
        }, inplace=True)

        display["Previous Month Diff"] = display["Previous Month Diff"] \
            .map("${:,.0f}".format)
        display["Current YTD Diff"] = display["Current YTD Diff"] \
            .map("${:,.0f}".format)
        display["Pct Improvement"] = display["Pct Improvement"] \
            .map("{:+.1f}%".format)

        disp_for_streamlit = display.copy()
        disp_for_streamlit["Date"] = disp_for_streamlit["Date"].dt.strftime("%Y-%m-%d")

        st.dataframe(disp_for_streamlit, use_container_width=True)
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

with tab4:
    st.header("4) YTD Difference Clustering")
    st.text("To define K-mean clustering, we have to understand each project's progression as one data point. We use each projects year-to-date difference to evaluate the project similarities in terms of path. This would allow us to group together different projects to further understand the patterns in the data. A cluster can be defined as one of these groups, with the number of clusters or groups chosen based on how well the data can be segmented. Feel free to explore the data further by downloading an individual cluster into an Excel file.")

    st.markdown("""
         • Same filters as before, with a user-controlled IQR multiplier for outlier detection (use slider above)  
         • Short-term (≤4 stages) vs Long-term (≥5 stages) clustering  
         • Download any cluster to Excel; excluded outlier rows shown at bottom
       """)

    port_mon = st.session_state.get("port_mon")
    if port_mon is None:
        st.warning("Please complete Tab 1 first.")
        st.stop()

    df = port_mon.copy()
    df["YTD_diff"] = df["YTD Savings"] - df["YTD Plan"]

    df = df.loc[
        (df["Outcome Status"] != "Gray") &
        (df["Execution Status"] != "Gray") &
        (~df["LOB"].isin(["xLOB", "Service"]))
        ].reset_index(drop=True)

    df.dropna(subset=["YTD_diff"], inplace=True)
    if df.empty:
        st.warning("No data after dropping missing-financial rows.")
        st.stop()

    iqr_mult = st.slider(
        "IQR multiplier for outlier detection (Larger the IQR, the more projects excluded)",
        min_value=0.5, max_value=5.0, value=3.0, step=0.5, key="tab4_iqr_mult"
    )
    Q1, Q3 = df["YTD_diff"].quantile([.25, .75])
    IQR = Q3 - Q1
    lower, upper = Q1 - iqr_mult * IQR, Q3 + iqr_mult * IQR
    outliers = df[(df["YTD_diff"] < lower) | (df["YTD_diff"] > upper)]
    df_main = df[(df["YTD_diff"] >= lower) & (df["YTD_diff"] <= upper)]
    st.write(f"Excluded {len(outliers)} row(s) with |YTD_diff| outside ±{iqr_mult}×IQR.")

    df_main["Year"] = df_main["Date"].dt.year.astype(str)
    df_main["Stage Level"] = df_main.sort_values(["Initiative/Epic", "Date"]) \
        .groupby("Initiative/Epic") \
        .cumcount() \
        .add(1)

    max_stage = df_main.groupby("Initiative/Epic")["Stage Level"].max()
    groups = {
        "Short (<=4 stages)": max_stage[max_stage <= 4].index,
        "Long  (>=5 stages)": max_stage[max_stage >= 5].index
    }

    for title, proj_ids in groups.items():
        with st.expander(f"{title} ({len(proj_ids)} projects)", expanded=True):
            if len(proj_ids) < 4:
                st.info("Need at least 4 projects to cluster.")
                continue

            sub = df_main[df_main["Initiative/Epic"].isin(proj_ids)].copy()

            wide = (
                sub
                .groupby(["Initiative/Epic", "Stage Level"])["YTD_diff"]
                .mean()
                .unstack(fill_value=0)
            )
            X = StandardScaler().fit_transform(wide.values)

            n = len(proj_ids)
            if n == 4:
                st.info("Exactly 4 projects, forcing k=4 (no silhouette).")
                k_opt = 4
            else:
                max_k = min(8, n - 1)
                sil_scores = {}
                for k in range(4, max_k + 1):
                    labels = KMeans(n_clusters=k, n_init=25,
                                    random_state=123).fit_predict(X)
                    sil_scores[k] = silhouette_score(X, labels)
                k_opt = max(sil_scores, key=sil_scores.get)
                st.write(f"Optimal k={k_opt}  (silhouette={sil_scores[k_opt]:.3f})")

            km = KMeans(n_clusters=k_opt, n_init=50,
                        random_state=123).fit(X)
            labs = pd.Series(km.labels_, index=wide.index, name="cluster")
            merged = sub.merge(labs, left_on="Initiative/Epic",
                               right_index=True)

            summary = (
                merged
                .groupby(["cluster", "Stage Level"])["YTD_diff"]
                .agg(mean="mean",
                     q25=lambda x: x.quantile(.25),
                     q75=lambda x: x.quantile(.75))
                .reset_index()
            )
            counts = (
                merged[["Initiative/Epic", "cluster"]]
                .drop_duplicates()
                .cluster.value_counts()
                .sort_index()
            )

            fig, ax = plt.subplots(figsize=(7, 4))
            for cl in counts.index:
                d = summary[summary["cluster"] == cl]
                ax.fill_between(d["Stage Level"], d["q25"], d["q75"], alpha=0.2)
                ax.plot(d["Stage Level"], d["mean"],
                        marker="o", lw=2,
                        label=f"Cluster {cl} (n={counts[cl]})")
            ax.set_title(f"{title}: Mean YTD_diff by Stage (k={k_opt})")
            ax.set_xlabel("Stage Level")
            ax.set_ylabel("Difference ($)")
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda v, _: f"${v:,.0f}")
            )
            ax.legend(bbox_to_anchor=(1, 1), loc="upper left", title="Cluster")
            st.pyplot(fig)

            sel = st.selectbox("Inspect cluster:", counts.index,
                               key=f"tab4_select_{title}")
            detail = merged[merged["cluster"] == sel] \
                .sort_values(["Initiative/Epic", "Stage Level"])
            st.write(f"Rows in cluster {sel}: {len(detail)}")
            disp_detail = detail.copy()
            disp_detail["Date"] = disp_detail["Date"].dt.strftime("%Y-%m-%d")

            st.dataframe(disp_detail, use_container_width=True)

            csv = detail.to_csv(index=False).encode("utf-8")
            st.download_button(
                label=f"Download Cluster {sel} to Excel",
                data=csv,
                file_name=f"cluster_{title.replace(' ', '_')}_{sel}.csv",
                mime="text/csv"
            )

            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sns.boxplot(
                x="Stage Level", y="YTD_diff",
                data=detail, palette="Set2",
                showfliers=False, ax=ax2
            )
            ax2.set_title(f"Cluster {sel}: YTD_diff by Stage")
            ax2.set_xlabel("Stage Level")
            ax2.set_ylabel("Difference ($)")
            ax2.yaxis.set_major_formatter(
                FuncFormatter(lambda v, _: f"${v:,.0f}")
            )
            st.pyplot(fig2)

    if not outliers.empty:
        st.subheader(f"Excluded Outlier Rows (|YTD_diff| > {iqr_mult}×IQR)")
        out = outliers[["Initiative/Epic", "Date", "LOB", "YTD_diff"]].copy()
        out["YTD_diff"] = out["YTD_diff"].map("${:,.0f}".format)
        disp_out = out.copy()
        disp_out["Date"] = disp_out["Date"].dt.strftime("%Y-%m-%d")

        st.dataframe(disp_out, use_container_width=True)
with tab5:
    st.header("5) YTD % Improvement Clustering")
    st.text(
        "Exact same filters & charts as performance clusters, but plotting the true % change "
        "between consecutive periods (dropping each project’s first period)."
    )
    st.markdown("""
      • Gray statuses & xLOB/Service filtered  
      • Compute `YTD_diff = YTD Savings – YTD Plan`  
      • Compute consecutive‐period % change (`Pct_Improve`) and drop each project’s first record  
      • Apply user-controlled IQR multiplier for outlier detection (use slider above)  
      • Short‐term (≤4 stages) vs Long‐term (≥5 stages) clustering  
      • Download any cluster to Excel; excluded outlier rows listed at bottom
    """)


    pm = st.session_state.get("port_mon")
    if pm is None:
        st.warning("Tab 1 must be completed first.")
        st.stop()
    df = pm.copy()
    df["YTD_diff"] = df["YTD Savings"] - df["YTD Plan"]
    df = df.loc[
        (df["Outcome Status"]   != "Gray") &
        (df["Execution Status"] != "Gray") &
        (~df["LOB"].isin(["xLOB","Service"]))
    ].reset_index(drop=True)

    df.dropna(subset=["YTD_diff"], inplace=True)
    if df.empty:
        st.warning("No data after dropping missing-financial rows.")
        st.stop()

    df = df.sort_values(["Initiative/Epic","Date"])
    df["prev"] = df.groupby("Initiative/Epic")["YTD_diff"].shift(1)
    df["Pct_Improve"] = (df["YTD_diff"] - df["prev"]) \
                        / df["prev"].abs() * 100
    df = (
        df
        .dropna(subset=["prev","Pct_Improve"])
        .replace([np.inf,-np.inf], np.nan)
        .dropna(subset=["Pct_Improve"])
        .reset_index(drop=True)
    )
    if df.empty:
        st.warning("No data after dropping first-period rows.")
        st.stop()

    iqr_mult = st.slider(
        "IQR multiplier for % improvement outliers (Larger the IQR, the more projects excluded)",
        min_value=0.5, max_value=5.0, value=3.0, step=0.5, key="tab5_iqr_mult"
    )
    q1, q3 = df["Pct_Improve"].quantile([.25, .75])
    iqr = q3 - q1
    lower, upper = q1 - iqr_mult * iqr, q3 + iqr_mult * iqr
    outliers = df[(df["Pct_Improve"] < lower) | (df["Pct_Improve"] > upper)]
    df_main = df[(df["Pct_Improve"] >= lower) & (df["Pct_Improve"] <= upper)]
    st.write(f"Excluded {len(outliers)} rows with |Pct_Improve| outside ±{iqr_mult}×IQR.")

    df_main["Stage Level"] = (
        df_main
        .sort_values(["Initiative/Epic","Date"])
        .groupby("Initiative/Epic")
        .cumcount()
        .add(1)
    )

    max_stage = df_main.groupby("Initiative/Epic")["Stage Level"].max()
    groups = {
      "Short (≤4 stages)": max_stage[max_stage<=4].index,
      "Long (≥5 stages)" : max_stage[max_stage>=5].index
    }

    for title, ids in groups.items():
        with st.expander(f"{title} ({len(ids)} projects)", expanded=True):
            if len(ids)<4:
                st.info("Need ≥4 projects to cluster."); continue

            sub = df_main[df_main["Initiative/Epic"].isin(ids)]

            wide = (
                sub
                .groupby(["Initiative/Epic","Stage Level"])["Pct_Improve"]
                .mean()
                .unstack(fill_value=0)
            )
            X = StandardScaler().fit_transform(wide.values)

            n = len(ids)
            if n == 4:
                st.info("Exactly 4 projects → forcing k=4.")
                k_opt = 4
            else:
                sil = {}
                max_k = min(8, n-1)
                for k in range(4, max_k+1):
                    labs = KMeans(n_clusters=k, n_init=25,
                                  random_state=123).fit_predict(X)
                    sil[k] = silhouette_score(X, labs)
                k_opt = max(sil, key=sil.get)
                st.write(f"Optimal k={k_opt} (sil={sil[k_opt]:.3f})")

            km     = KMeans(n_clusters=k_opt, n_init=50,
                            random_state=123).fit(X)
            labs   = pd.Series(km.labels_, index=wide.index, name="cluster")
            merged = sub.merge(labs, left_on="Initiative/Epic",
                               right_index=True)

            summary = (
                merged
                .groupby(["cluster","Stage Level"])["Pct_Improve"]
                .agg(mean="mean",
                     q25=lambda x: x.quantile(.25),
                     q75=lambda x: x.quantile(.75))
                .reset_index()
            )
            counts = (
                merged[["Initiative/Epic","cluster"]]
                .drop_duplicates()
                .cluster
                .value_counts()
                .sort_index()
            )

            fig, ax = plt.subplots(figsize=(7,4))
            for cl in counts.index:
                d = summary[summary["cluster"]==cl]
                ax.fill_between(d["Stage Level"], d["q25"], d["q75"], alpha=0.2)
                ax.plot(d["Stage Level"], d["mean"],
                        marker="o", lw=2,
                        label=f"Cluster {cl} (n={counts[cl]})")
            ax.set_title(f"{title}: Mean % Improvement by Stage (k={k_opt})")
            ax.set_xlabel("Stage Level")
            ax.set_ylabel("Pct Improvement")
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_: f"{v:.1f}%"))
            ax.legend(bbox_to_anchor=(1,1), loc="upper left", title="Cluster")
            st.pyplot(fig)

            sel = st.selectbox("Inspect cluster:", counts.index,
                               key=f"tab5_select_{title}")
            detail = merged[merged["cluster"]==sel] \
                         .sort_values(["Initiative/Epic","Stage Level"])
            st.write(f"Rows in cluster {sel}: {len(detail)}")

            disp_detail = detail[[
                "Initiative/Epic", "Date", "LOB", "prev", "Pct_Improve"
            ]].copy()
            disp_detail["Date"] = disp_detail["Date"].dt.strftime("%Y-%m-%d")

            st.dataframe(disp_detail, use_container_width=True)

            csv = detail.to_csv(index=False).encode("utf-8")
            st.download_button(
                f"Download Cluster {sel} to Excel",
                data=csv,
                file_name=f"{title.replace(' ','_')}_cluster_{sel}.csv",
                mime="text/csv"
            )

            fig2, ax2 = plt.subplots(figsize=(6,4))
            sns.boxplot(
                x="Stage Level", y="Pct_Improve",
                data=detail, palette="Set2",
                showfliers=False, ax=ax2
            )
            ax2.set_title(f"Cluster {sel}: % Improvement by Stage")
            ax2.set_xlabel("Stage Level")
            ax2.set_ylabel("Pct Improvement")
            ax2.yaxis.set_major_formatter(FuncFormatter(lambda v,_: f"{v:.1f}%"))
            st.pyplot(fig2)

    if not outliers.empty:
        st.subheader(f"Excluded Outliers (|Pct_Improve| > {iqr_mult}×IQR)")

        st.markdown(
            """
            **Field definitions:**  
            • **Previous Month Diff**: the prior period’s YTD difference  
              (YTD Savings – YTD Plan).  
            • **Pct Improvement**: percentage change from Previous Month Diff  
              to the current YTD difference, calculated as  
              100 × (Current – Previous) / |Previous|.
            """
        )

        display = outliers[[
            "Initiative/Epic", "Date", "LOB", "prev", "Pct_Improve"
        ]].copy().rename(columns={
            "prev": "Previous Month Diff",
            "Pct_Improve": "Pct Improvement"
        })

        display["Previous Month Diff"] = display["Previous Month Diff"] \
            .map("${:,.0f}".format)
        display["Pct Improvement"] = display["Pct Improvement"] \
            .map("{:+.1f}%".format)

        disp_out = display.copy()
        disp_out["Date"] = disp_out["Date"].dt.strftime("%Y-%m-%d")

        st.dataframe(disp_out, use_container_width=True)

with tab6:
    st.header("6) Comment Sentiment & Word Cloud")
    st.markdown("""
    We unify Execution, Outcome and Path‐to‐Green comments into one field,  
    dedupe repeats, score polarity from –1 (very negative) to +1 (very positive),  
    and explore sentiment by month & status.  
    Then generate a WordCloud (custom stopwords removed) and click any word  
    to see the original comment(s).
    """)

    pm = st.session_state.get("port_mon")
    if pm is None:
        st.warning("Complete Tab 1 first.")
        st.stop()

    df = pm.copy().loc[
        (pm["Execution Status"] != "Gray") &
        (pm["Outcome Status"]   != "Gray") &
        (~pm["LOB"].isin(["xLOB","Service"]))
    ].reset_index(drop=True)

    df["Comments"] = (
        df["Execution Comments"].fillna("") + " ∥ " +
        df["Outcome Comments"].fillna("")   + " ∥ " +
        df["Path to Green"].fillna("")
    )
    df = df[df["Comments"].str.strip().astype(bool)]
    df = df.drop_duplicates(subset="Comments").reset_index(drop=True)

    df["Sentiment"] = df["Comments"].apply(lambda t: TextBlob(t).sentiment.polarity)

    df["Period"] = df["Date"].dt.to_period("M")
    periods = sorted(df["Period"].unique())

    start, end = st.select_slider(
        "Select Month Range:",
        options=periods,
        value=(periods[0], periods[-1]),
        format_func=lambda p: p.strftime("%b %Y"),
        key="sent_month_range"
    )
    df = df[(df["Period"] >= start) & (df["Period"] <= end)]
    if df.empty:
        st.warning("No comments in that month range.")
        st.stop()

    df["Month"] = df["Date"].dt.month_name().str[:3]
    month_order = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]

    st.subheader("Execution Comments: Yellow vs Red")
    exe = df[df["Execution Status"].isin(["Yellow","Red"])]
    if exe.empty:
        st.info("No Execution comments for Yellow/Red in this range.")
    else:
        plt.figure(figsize=(10,4))
        sns.violinplot(
            x="Month", y="Sentiment", hue="Execution Status", data=exe,
            split=True, inner="quartile",
            palette={"Yellow":"gold","Red":"firebrick"},
            order=[m for m in month_order if m in exe["Month"].unique()]
        )
        plt.xticks(rotation=30)
        plt.ylabel("Polarity (–1…+1)")
        plt.xlabel("Month")
        plt.title("Execution Comment Sentiment by Month & Status")
        st.pyplot(plt.gcf(), use_container_width=True)

    # 7) Outcome sentiment violins
    st.subheader("Outcome Comments: Yellow vs Red")
    out = df[df["Outcome Status"].isin(["Yellow","Red"])]
    if out.empty:
        st.info("No Outcome comments for Yellow/Red in this range.")
    else:
        plt.figure(figsize=(10,4))
        sns.violinplot(
            x="Month", y="Sentiment", hue="Outcome Status", data=out,
            split=True, inner="quartile",
            palette={"Yellow":"gold","Red":"firebrick"},
            order=[m for m in month_order if m in out["Month"].unique()]
        )
        plt.xticks(rotation=30)
        plt.ylabel("Polarity (–1…+1)")
        plt.xlabel("Month")
        plt.title("Outcome Comment Sentiment by Month & Status")
        st.pyplot(plt.gcf(), use_container_width=True)

    st.subheader("Word Cloud of Comments by LOB")
    lob_opts = sorted(df["LOB"].unique())
    lob_sel  = st.multiselect(
        "Include LOB(s):", options=lob_opts,
        default=lob_opts, key="lob_wordcloud"
    )
    if not lob_sel:
        st.info("Select at least one LOB to generate the word cloud.")
    else:
        text = " ".join(df[df["LOB"].isin(lob_sel)]["Comments"].tolist())
        if not text.strip():
            st.warning("No comments for selected LOB(s).")
        else:
            custom_stop = set(STOPWORDS) | {
                "to","the","and","in","of","for","with","is","we",
                "on","be","a","are","will","this","asat","by","not",
                "at","so","can","has"
            }
            wc = WordCloud(
                width=1200, height=600,
                background_color="white",
                stopwords=custom_stop,
                max_words=200,
                contour_width=1, contour_color="steelblue"
            ).generate(text)

            img = wc.to_image()
            st.image(img, use_container_width=True)

            # 9) Clickable word‐filter
            words = list(wc.words_.keys())
            sel_words = st.multiselect(
                "Click words to filter comments:", options=words,
                key="word_filter"
            )
            if sel_words:
                pattern = "|".join(map(re.escape, sel_words))
                df_ctx = df[df["Comments"].str.contains(pattern, case=False)]
                st.subheader("Comments containing: " + ", ".join(sel_words))

                disp_ctx = df_ctx.copy()
                disp_ctx["Date"] = disp_ctx["Date"].dt.strftime("%Y-%m-%d")
                disp_ctx["Sentiment"] = disp_ctx["Sentiment"].map("{:+.2f}".format)

                st.dataframe(
                    disp_ctx[[
                        "Date", "Initiative/Epic", "LOB",
                        "Execution Status", "Outcome Status",
                        "Sentiment", "Comments"
                    ]],
                    use_container_width=True
                )
    if st.checkbox("Show scored comments table", key="show_comments_table"):
        table = df[[
            "Date", "Initiative/Epic", "LOB",
            "Execution Status", "Outcome Status",
            "Sentiment", "Comments"
        ]].copy()

        table["Date"] = table["Date"].dt.strftime("%Y-%m-%d")
        table["Sentiment"] = table["Sentiment"].map("{:+.2f}".format)

        st.dataframe(table, use_container_width=True)