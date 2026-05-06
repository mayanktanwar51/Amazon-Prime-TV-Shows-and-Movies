# Import Libraries
import numpy as np
import pandas as pd
import ast
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("Prepared_AMprime.zip", compression='zip')

Amazon_Prime = load_data()

@st.cache_data
def prepare_category_metric(df, col, value_col, top_n=10):
    data = df.copy()
    data[col] = data[col].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    data = data.explode(col)
    data = data[data[col].notna() & (data[col] != "")]

    top = data[col].value_counts().head(top_n).index
    data = data[data[col].isin(top)]

    agg = (
        data.groupby([col, 'type'])[value_col]
        .agg(
            Mean='mean',
            Q1=lambda x: x.quantile(0.25),
            Q3=lambda x: x.quantile(0.75)
        )
        .reset_index()
    )
    agg['Upper'] = agg['Q3'] + 1.5 * (agg['Q3'] - agg['Q1'])

    agg = agg.melt(
        id_vars=[col, 'type'],
        value_vars=['Mean', 'Q3', 'Upper'],
        var_name='Metric',
        value_name='Score'
    )
    agg['Label'] = agg['type'] + ' | ' + agg['Metric']

    order = (
        data.groupby(col)[value_col]
        .mean()
        .sort_values(ascending=False)
        .index
    )

    return agg, order


def plot_metric(df, value_col, title_name):
    country_df, country_order = prepare_category_metric(
        df, 'production_countries', value_col
    )
    genre_df, genre_order = prepare_category_metric(df, 'genres', value_col)

    colors = {
        'MOVIE | Mean':  '#37E8C1',
        'MOVIE | Q3':    '#B03CE8',
        'MOVIE | Upper': '#3588E4',
        'SHOW | Mean':   '#DE9632',
        'SHOW | Q3':     '#0C9E26',
        'SHOW | Upper':  '#EEB182'
    }

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Production Countries", "Genres"),
        horizontal_spacing=0.12
    )

    for label in country_df['Label'].unique():
        d1 = country_df[country_df['Label'] == label]
        fig.add_bar(
            x=d1['Score'],
            y=d1['production_countries'],
            orientation='h',
            name=label,
            marker=dict(color=colors[label], line=dict(color='white', width=0.5)),
            width=0.22,
            legendgroup=label,
            showlegend=True,
            row=1, col=1
        )

        d2 = genre_df[genre_df['Label'] == label]
        fig.add_bar(
            x=d2['Score'],
            y=d2['genres'],
            orientation='h',
            name=label,
            marker=dict(color=colors[label], line=dict(color='white', width=0.5)),
            width=0.22,
            legendgroup=label,
            showlegend=False,
            row=1, col=2
        )

    fig.update_yaxes(
        categoryorder='array',
        categoryarray=country_order,
        row=1, col=1
    )
    fig.update_yaxes(
        categoryorder='array',
        categoryarray=genre_order,
        row=1, col=2
    )

    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor='#0F171E',
        paper_bgcolor='#0F171E',
        font=dict(color='white'),
        title=f"{title_name} Analysis Bottom (Mean vs Q3 vs Upper Bound)",
        title_x=0.5,
        barmode='group',
        bargap=0.55,
        bargroupgap=0.3,
        height=750,
        width=1100,
        legend=dict(
            orientation='h',
            y=-0.18,
            x=0.5,
            xanchor='center'
        )
    )

    return fig

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------

st.set_page_config(
    page_title="Amazon Prime OTT Insights",
    page_icon="🎬",
    layout="wide"
)

# ------------------------------------------------------------
# CUSTOM CSS
# ------------------------------------------------------------

st.markdown(r"""
<style>

.stApp {
    background:
        linear-gradient(rgba(0,0,0,0.70), rgba(0,0,0,0.70)),
        url("https://images.unsplash.com/photo-1489599849927-2ee91cede3ba");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

.logo-container {
    display: flex;
    justify-content: center;
    margin-top: 10px;
}

.prime-logo {
    width: 320px;
}

h1, h2, h3, p, li {
    color: white !important;
}

.glass-card {
    background: rgba(255,255,255,0.08);
    padding: 25px;
    border-radius: 18px;
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255,255,255,0.15);
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# SIDEBAR CSS
# ------------------------------------------------------------

st.markdown("""
<style>

section[data-testid="stSidebar"] {
    background: rgba(90, 90, 90, 0.22);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border-right: 1px solid rgba(255,255,255,0.08);
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

section[data-testid="stSidebar"] h1 {
    color: white !important;
    font-size: 28px !important;
    font-weight: 700 !important;
}

.stRadio label {
    font-size: 17px !important;
    font-weight: 500 !important;
}

div[data-baseweb="select"] > div {
    background: rgba(120,120,120,0.18) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 14px !important;
    backdrop-filter: blur(10px);
}

div[data-baseweb="select"] span {
    color: white !important;
}

div[data-baseweb="popover"] ul {
    background: rgba(70, 70, 70, 0.95) !important;
    backdrop-filter: blur(15px) !important;
    -webkit-backdrop-filter: blur(15px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
    padding: 10px 0px !important;
}

div[data-baseweb="popover"] li {
    background-color: transparent !important;
    color: white !important;
    font-weight: 500 !important;
}

div[data-baseweb="popover"] li:hover {
    background-color: rgba(255, 255, 255, 0.1) !important;
}

div[data-baseweb="popover"] li[aria-selected="true"] {
    background-color: rgba(255, 255, 255, 0.15) !important;
    color: white !important;
}

div[data-baseweb="popover"] > div {
    background-color: transparent !important;
    border: none !important;
}

li:hover {
    background: rgba(255,255,255,0.08) !important;
}

hr {
    border: 1px solid rgba(255,255,255,0.08);
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# SIDEBAR LOGO
# ------------------------------------------------------------

col1, col2, col3 = st.sidebar.columns([1, 3, 1])

with col2:
    st.image(
        "amazon-prime-video-logo-png_seeklogo-476889.png",
        use_container_width=True
    )

st.sidebar.markdown("---")

# ------------------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------------------

st.sidebar.title("Navigation")

main_menu = st.sidebar.radio(
    "Select Section",
    [
        "Home Page",
        "Graph Matrixs",
        "Insights Q & A",
        "Conclusion"
    ]
)

# ============================================================
# HOME PAGE
# ============================================================

if main_menu == "Home Page":

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.image(
            "amazon-prime-video-2022-logo.png",
            width=380
        )

    st.markdown(
        "<h1 style='text-align:center;'>Amazon Prime OTT Insights</h1>",
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="glass-card">

    <h3>Introduction</h3>

    <p>
    The entertainment industry has experienced a major transformation with the rise of
    OTT (Over-The-Top) streaming platforms such as Amazon Prime Video. Millions of users
    worldwide consume Movies and TV Shows through digital platforms, generating massive
    amounts of content-related data. Understanding this data helps businesses identify
    viewer preferences, content trends, audience engagement, and platform growth opportunities.
    </p>

    <p>
    This interactive dashboard is designed to analyze Amazon Prime content using
    Exploratory Data Analysis (EDA). The application provides insights into genre
    distribution, release trends, ratings, popularity, runtime patterns, and
    country-wise content availability. Users can compare Movies and TV Shows,
    discover top-performing categories, and explore relationships between
    ratings, popularity, and content quality.
    </p>

    <p>
    Using Python, Streamlit, Plotly, and Seaborn, the project delivers a
    modern, interactive, and business-oriented analytics experience that supports
    data-driven decision-making for OTT platforms and content strategy planning.
    </p>

    <h3>🎯 Project Objectives</h3>

    <ul>
    <li>Analyze Amazon Prime content library structure</li>
    <li>Identify top-performing genres and categories</li>
    <li>Study trends in Movies and TV Shows over time</li>
    <li>Explore ratings, popularity, and audience preferences</li>
    <li>Compare Movie vs TV Show performance</li>
    <li>Generate business-focused OTT insights</li>
    <li>Build interactive and professional visualizations</li>
    <li>Create an end-to-end EDA dashboard using Streamlit</li>
    </ul>

    </div>
    """, unsafe_allow_html=True)

# ============================================================
# GRAPH MATRIXS
# ============================================================

elif main_menu == "Graph Matrixs":

    st.title("Graph Matrixs")

    graph_menu = st.sidebar.selectbox(
        "Choose Matrix Type",
        [
            "Univariate Matrix",
            "Bivariate Matrix",
            "Multivariate Matrix"
        ]
    )

    # --------------------------------------------------------
    # UNIVARIATE MATRIX
    # --------------------------------------------------------

    if graph_menu == "Univariate Matrix":

        uni_option = st.sidebar.radio(
            "Select Variable Type",
            [
                "Numeric Variable",
                "Categorical Variable",
                "Genres and Production Countries"
            ]
        )

        st.header("Univariate Matrix")

        # ----------------------------------------------------
        # NUMERIC VARIABLE
        # ----------------------------------------------------

        if uni_option == "Numeric Variable":

            st.markdown("""
            <div class="glass-card">
                <h2>Numeric Variable Distribution</h2>
            </div>
            """, unsafe_allow_html=True)

            num_cols = [
                'imdb_score',
                'tmdb_score',
                'imdb_votes',
                'tmdb_popularity',
                'runtime'
            ]

            colors = {
                'MOVIE': '#00A8E1',
                'SHOW': '#FF9900'
            }

            fig = make_subplots(
                rows=2,
                cols=3,
                subplot_titles=num_cols,
                horizontal_spacing=0.08,
                vertical_spacing=0.12
            )

            positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)]

            for (col, pos) in zip(num_cols, positions):

                r, c = pos

                for t in ['MOVIE', 'SHOW']:

                    d = Amazon_Prime[Amazon_Prime['type'] == t]

                    fig.add_trace(
                        go.Histogram(
                            x=d[col],
                            name=t,
                            marker_color=colors[t],
                            opacity=0.60,
                            showlegend=(col == num_cols[0])
                        ),
                        row=r,
                        col=c
                    )

            fig.update_layout(
                template='plotly_dark',
                plot_bgcolor='#0F171E',
                paper_bgcolor='#0F171E',
                font=dict(color='white'),
                height=700,
                title="Distribution of Numerical Features",
                title_x=0.5,
                barmode='overlay',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                )
            )

            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(gridcolor='#2A3A4A')

            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            <div class="glass-card">

            <h3>Key Insights</h3>

            <ul>

            <li><b>IMDb Score & TMDb Score:</b>
            <ul>
            <li>Movie scores are approximately normally distributed with an average score around <b>6</b>.</li>
            <li>TV Shows show a slightly left-skewed distribution with a higher average score of nearly <b>7.5</b>.</li>
            <li>TV Shows generally receive better audience ratings compared to Movies.</li>
            </ul>
            </li>

            <li><b>IMDb Votes & TMDb Popularity:</b>
            <ul>
            <li>Both variables are heavily right-skewed.</li>
            <li>Most Movies and TV Shows have low popularity and fewer audience votes.</li>
            <li>The number of titles decreases sharply as popularity and votes increase.</li>
            </ul>
            </li>

            <li><b>Runtime Analysis:</b>
            <ul>
            <li>Movies are slightly left-skewed with an average runtime around <b>90 minutes</b>.</li>
            <li>TV Shows display a bimodal distribution, where runtimes around <b>24 minutes</b> and <b>45 minutes</b> are most common.</li>
            <li>Standard runtime content dominates the platform.</li>
            </ul>
            </li>

            </ul>

            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="glass-card">

            <h3>Business Interpretation</h3>

            <ul>

            <li><b>IMDb Score & TMDb Score:</b>
            <ul>
            <li>Movies maintain a balanced score distribution, indicating stable audience engagement.</li>
            <li>TV Shows perform better overall, suggesting users currently prefer episodic and binge-watchable content.</li>
            <li>Higher TV Show ratings indicate strong growth opportunities in premium series production.</li>
            </ul>
            </li>

            <li><b>IMDb Votes & TMDb Popularity:</b>
            <ul>
            <li>Low popularity and voting activity indicate limited audience reach for most titles.</li>
            <li>Amazon Prime may need stronger recommendation systems, promotions, and marketing strategies to improve audience engagement.</li>
            <li>Only a small percentage of content achieves high visibility and audience traction.</li>
            </ul>
            </li>

            <li><b>Runtime Analysis:</b>
            <ul>
            <li>Movies with runtimes between <b>90–100 minutes</b> appear to be the preferred format.</li>
            <li>TV Shows around <b>45 minutes</b> align with industry-standard episode duration and viewer retention patterns.</li>
            <li>Longer runtimes can still perform well when supported by strong storytelling and audience engagement.</li>
            </ul>
            </li>

            <li><b>Overall Observation:</b>
            <ul>
            <li>Numerical distributions provide a strong overview of audience behavior, but deeper analysis is required to identify the factors behind high-performing content.</li>
            </ul>
            </li>

            </ul>

            </div>
            """, unsafe_allow_html=True)

        # ----------------------------------------------------
        # CATEGORICAL VARIABLE
        # ----------------------------------------------------

        elif uni_option == "Categorical Variable":

            st.markdown("""
            <div class="glass-card">
                <h2>Categorical Variable Dashboard</h2>
            </div>
            """, unsafe_allow_html=True)

            df = Amazon_Prime.copy()

            role_df = df.copy()
            role_df['role'] = (
                role_df['role']
                .fillna('')
                .str.split(',')
            )
            role_df = role_df.explode('role')
            role_df['role'] = role_df['role'].str.strip()

            fig = make_subplots(
                rows=2,
                cols=2,
                specs=[
                    [{"type": "domain"}, {"type": "xy"}],
                    [{"type": "xy"}, {"type": "domain"}]
                ],
                subplot_titles=(
                    "Era by Type",
                    "Content Quality by Type",
                    "Runtime Category by Type",
                    "Top Roles by Type"
                ),
                vertical_spacing=0.15,
                horizontal_spacing=0.12
            )

            # ERA DONUT
            era = (
                df.groupby(['Era', 'type'])
                .size()
                .reset_index(name='count')
            )

            for t in era['type'].unique():
                temp = era[era['type'] == t]
                fig.add_trace(
                    go.Pie(
                        labels=temp['Era'],
                        values=temp['count'],
                        hole=0.55,
                        name=t
                    ),
                    row=1, col=1
                )

            # CONTENT QUALITY
            cq = (
                df.groupby(['Content_Quality', 'type'])
                .size()
                .reset_index(name='count')
            )

            bar1 = px.bar(
                cq,
                x='Content_Quality',
                y='count',
                color='type',
                barmode='group',
                color_discrete_map={
                    'MOVIE': '#00A8E1',
                    'SHOW': '#FF9900'
                }
            )

            for tr in bar1.data:
                fig.add_trace(tr, row=1, col=2)

            # RUNTIME CATEGORY
            rt = (
                df.groupby(['runtime_category', 'type'])
                .size()
                .reset_index(name='count')
            )

            bar2 = px.bar(
                rt,
                x='runtime_category',
                y='count',
                color='type',
                barmode='group',
                color_discrete_map={
                    'MOVIE': '#00A8E1',
                    'SHOW': '#FF9900'
                }
            )

            for tr in bar2.data:
                fig.add_trace(tr, row=2, col=1)

            # ROLE DONUT
            top_roles = role_df['role'].value_counts().head(5).index

            role = (
                role_df[role_df['role'].isin(top_roles)]
                .groupby(['role', 'type'])
                .size()
                .reset_index(name='count')
            )

            for t in role['type'].unique():
                temp = role[role['type'] == t]
                fig.add_trace(
                    go.Pie(
                        labels=temp['role'],
                        values=temp['count'],
                        hole=0.55,
                        name=t
                    ),
                    row=2, col=2
                )

            fig.update_layout(
                height=700,
                template='plotly_dark',
                plot_bgcolor='#0F171E',
                paper_bgcolor='#0F171E',
                font=dict(color='white'),
                title='Categorical Variable Dashboard',
                title_x=0.5
            )

            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            <div class="glass-card">

            <h3>Key Insights</h3>

            <ul>
            <li>Modern Era content dominates Amazon Prime and contributes nearly half of the total content library.</li>
            <li>Most titles fall under the <b>"Poor"</b> content quality category, especially Movies.</li>
            <li>Standard Movies represent the most common runtime category on the platform.</li>
            <li>TV Shows are primarily concentrated in <b>Short Episode</b> and <b>Standard Episode</b> runtime groups.</li>
            <li><b>ACTOR</b> roles dominate the dataset, while <b>DIRECTOR</b> roles account for a comparatively smaller share.</li>
            <li>Movies significantly outnumber TV Shows across most categorical segments.</li>
            </ul>

            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="glass-card">

            <h3>Business Interpretation</h3>

            <h4>Positive Business Impact</h4>

            <ul>
            <li>Strong dominance of Modern Era content shows that Amazon Prime is aligned with current audience trends and modern viewing preferences.</li>
            <li>A larger Movie catalog helps attract diverse audience groups and increases overall platform engagement.</li>
            <li>Standard-length Movies and Episodes appear to match audience viewing habits and attention spans effectively.</li>
            </ul>

            <h4>Growth & Improvement Areas</h4>

            <ul>
            <li>The high concentration of content in the <b>"Poor"</b> quality category may negatively affect viewer satisfaction and retention.</li>
            <li>The lower number of TV Shows compared to Movies limits binge-watching opportunities and long-term platform engagement.</li>
            <li>Role distribution is highly concentrated around ACTOR roles, reducing the scope for deeper creator-based insights and personalization.</li>
            <li>Amazon Prime can improve content quality diversification and expand high-quality TV Show production to strengthen platform growth.</li>
            </ul>

            </div>
            """, unsafe_allow_html=True)

        # ----------------------------------------------------
        # GENRES AND PRODUCTION COUNTRIES
        # ----------------------------------------------------

        elif uni_option == "Genres and Production Countries":

            sub_option = st.radio(
                "Select Analysis",
                ["Genres", "Production Countries"],
                horizontal=True
            )

            # GENRES
            if sub_option == "Genres":

                Amazon_Prime['genres'] = Amazon_Prime['genres'].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
                )

                df = Amazon_Prime.explode('genres')
                df = df[df['genres'].notna()]

                genre_type_counts = (
                    df.groupby(['genres', 'type'])
                    .size()
                    .reset_index(name='Count')
                )

                total_counts = (
                    genre_type_counts.groupby('genres')['Count']
                    .sum()
                    .sort_values(ascending=False)
                )

                top_genres = total_counts.index[:15]

                genre_type_counts = genre_type_counts[
                    genre_type_counts['genres'].isin(top_genres)
                ]

                fig = px.bar(
                    genre_type_counts,
                    x='Count',
                    y='genres',
                    color='type',
                    orientation='h',
                    barmode='group',
                    title='Top Genres by Content Type',
                    text='Count',
                    color_discrete_map={
                        'MOVIE': '#00A8E1',
                        'SHOW': '#FF9900'
                    }
                )

                fig.update_layout(
                    template='plotly_dark',
                    plot_bgcolor='#0f171e',
                    paper_bgcolor='#0f171e',
                    font=dict(color='white'),
                    title_x=0.5,
                    height=650,
                    xaxis_title='Number of Titles',
                    yaxis_title='Genre'
                )

                fig.update_yaxes(categoryorder='total ascending')
                fig.update_traces(textposition='outside')

                st.plotly_chart(fig, use_container_width=True)

                st.markdown("""
                <div class="glass-card">

                <h3>Key Insights</h3>

                <ul>

                <li><b>Top Genres</b>
                <ul>
                <li><b>Movies:</b> Drama, Comedy, Thriller, Action, and Romance are the most dominant genres.</li>
                <li><b>TV Shows:</b> Drama, Comedy, Action, Sci-Fi, and Animation are the leading genres.</li>
                </ul>
                </li>

                <li><b>Average Performing Genres</b>
                <ul>
                <li><b>Movies:</b> European, Sci-Fi, Fantasy, Family, and Western genres show average performance.</li>
                <li><b>TV Shows:</b> Similar genre distribution is observed for TV Shows.</li>
                </ul>
                </li>

                <li><b>Lowest Performing Genres</b>
                <ul>
                <li><b>Movies:</b> Music, War, Animation, and Sports have the lowest representation.</li>
                <li><b>Reality</b> content is almost absent in Movies but exists in TV Shows.</li>
                <li><b>TV Shows:</b> Sports, Reality, War, Music, and History are among the least common genres.</li>
                </ul>
                </li>

                </ul>

                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                <div class="glass-card">

                <h3>Business Interpretation</h3>

                <ul>
                <li>Movies and TV Shows show relatively balanced genre diversity across the platform.</li>
                <li>Drama and Comedy dominate both categories, indicating strong audience preference for emotionally engaging and entertainment-focused content.</li>
                <li>TV Shows perform particularly well in genres such as Sci-Fi and Animation, suggesting increasing demand for binge-watchable niche content.</li>
                <li>Low-performing genres such as Sports, War, and Music may require stronger storytelling, marketing, or audience targeting strategies.</li>
                <li>Reality content shows stronger presence in TV Shows compared to Movies, highlighting a platform opportunity for unscripted episodic content.</li>
                <li>Further analysis combining genres with ratings, votes, and popularity metrics is required to identify the most commercially successful genres.</li>
                </ul>

                </div>
                """, unsafe_allow_html=True)

            # PRODUCTION COUNTRIES
            elif sub_option == "Production Countries":

                st.markdown("""
                <div class="glass-card">
                    <h2>Production Countries Analysis</h2>
                </div>
                """, unsafe_allow_html=True)

                Amazon_Prime['production_countries'] = (
                    Amazon_Prime['production_countries']
                    .apply(
                        lambda x: ast.literal_eval(x)
                        if isinstance(x, str) else x
                    )
                )

                df_exploded = Amazon_Prime.explode('production_countries')
                df_exploded = df_exploded[
                    df_exploded['production_countries'].notna()
                ]

                country_type_counts = (
                    df_exploded
                    .groupby(['production_countries', 'type'])
                    .size()
                    .unstack(fill_value=0)
                )

                country_type_counts['total'] = country_type_counts.sum(axis=1)
                country_type_counts = country_type_counts.sort_values(
                    by='total', ascending=False
                )

                top_20 = (
                    country_type_counts.iloc[:20]
                    .sort_values(by='total', ascending=True)
                )
                middle_20 = (
                    country_type_counts.iloc[58:78]
                    .sort_values(by='total', ascending=True)
                )
                bottom_20 = (
                    country_type_counts.iloc[96:]
                    .sort_values(by='total', ascending=True)
                )

                top_20 = top_20.drop(columns='total')
                middle_20 = middle_20.drop(columns='total')
                bottom_20 = bottom_20.drop(columns='total')

                fig = make_subplots(
                    rows=1,
                    cols=3,
                    subplot_titles=(
                        "Top 20 Countries",
                        "Average Countries",
                        "Bottom Countries"
                    ),
                    horizontal_spacing=0.05
                )

                def add_grouped_bar(data, col):
                    for t in data.columns:
                        fig.add_trace(
                            go.Bar(
                                x=data[t],
                                y=data.index,
                                orientation='h',
                                name=t
                            ),
                            row=1,
                            col=col
                        )

                add_grouped_bar(top_20, 1)
                add_grouped_bar(middle_20, 2)
                add_grouped_bar(bottom_20, 3)

                fig.update_layout(
                    title="Production Countries Distribution by Type",
                    barmode='group',
                    template='plotly_dark',
                    plot_bgcolor='#0f171e',
                    paper_bgcolor='#0f171e',
                    font=dict(color='white', size=12),
                    height=600
                )

                fig.update_yaxes(automargin=True, tickfont=dict(size=8))
                fig.update_xaxes(title_text="Number of Titles")

                st.plotly_chart(fig, use_container_width=True)

                st.markdown("""
                <div class="glass-card">

                <h3>Key Insights</h3>

                <ul>

                <li><b>Top Production Countries</b>
                <ul>
                <li><b>Movies:</b> USA, UK, and India are the leading movie production countries.</li>
                <li><b>TV Shows:</b> USA, UK, Canada, China, India, Japan, and Australia dominate TV Show production.</li>
                </ul>
                </li>

                <li><b>Average Production Countries</b>
                <ul>
                <li><b>Movies:</b> Ethiopia, Vatican City, and Egypt show average production contribution.</li>
                <li><b>TV Shows:</b> Singapore, Turkey, Egypt, and Uruguay represent average TV Show production activity.</li>
                </ul>
                </li>

                <li><b>Lowest Production Countries</b>
                <ul>
                <li><b>Movies:</b> Antarctica, Libya, and Puerto Rico contribute minimal movie production.</li>
                <li><b>TV Shows:</b> Around 55–57 countries show almost zero TV Show production activity.</li>
                </ul>
                </li>

                </ul>

                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                <div class="glass-card">

                <h3>Business Interpretation</h3>

                <h4>Strong Production Markets</h4>

                <ul>
                <li>Countries such as USA, UK, India, Canada, and China already have strong production ecosystems and consistent audience demand.</li>
                <li>These countries represent mature content markets that continuously generate high-volume OTT content.</li>
                </ul>

                <h4>Growth Opportunity Markets</h4>

                <ul>
                <li>Countries such as Singapore, Turkey, Egypt, Uruguay, and Ethiopia show moderate production activity and strong future growth potential.</li>
                <li>Investing in these regions may provide cost-efficient content expansion and new audience acquisition opportunities.</li>
                <li>Regional collaborations and localized storytelling can help these markets scale globally.</li>
                </ul>

                <h4>Low Production Markets</h4>

                <ul>
                <li>Countries with minimal or zero production activity may require significant investment, infrastructure, and audience development.</li>
                <li>Strategic collaborations with experienced actors, directors, and producers from developed content markets can help accelerate growth in these regions.</li>
                <li>Introducing local talent through supporting roles and regional productions may gradually improve audience engagement and production capacity.</li>
                </ul>

                </div>
                """, unsafe_allow_html=True)

    # --------------------------------------------------------
    # BIVARIATE MATRIX
    # --------------------------------------------------------

    elif graph_menu == "Bivariate Matrix":

        st.header("Bivariate Matrix")

        bi_option = st.sidebar.radio(
            "Select Analysis",
            [
                "AMprime Score Matrix",
                "AMprime Popularity Matrix",
                "AMprime Rating Matrix"
            ]
        )

        # ------------------------------------------------
        # REUSABLE FUNCTION
        # ------------------------------------------------

        def plot_multi_analysis(df, value_col, title_name):

            df = df.copy()

            era_df = (
                df.groupby(['Era', 'type'])[value_col]
                .mean()
                .reset_index()
            )

            runtime_df = (
                df.groupby(['runtime_category', 'type'])[value_col]
                .mean()
                .reset_index()
            )

            colors = {
                'MOVIE': '#00A8E1',
                'SHOW': '#FF9900'
            }

            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    f"Bar: {title_name} vs Era",
                    f"Box: {title_name} vs Content Quality",
                    f"Line: {title_name} vs Runtime Category",
                    f"Scatter: {title_name} vs Popularity"
                )
            )

            for t in ['MOVIE', 'SHOW']:

                d_main = df[df['type'] == t]
                d_era = era_df[era_df['type'] == t]
                d_run = runtime_df[runtime_df['type'] == t]

                # Bar
                fig.add_bar(
                    x=d_era['Era'],
                    y=d_era[value_col],
                    name=t,
                    marker_color=colors[t],
                    legendgroup=t,
                    showlegend=True,
                    row=1,
                    col=1
                )

                # Box
                fig.add_box(
                    x=d_main['Content_Quality'],
                    y=d_main[value_col],
                    name=t,
                    marker_color=colors[t],
                    legendgroup=t,
                    showlegend=False,
                    row=1,
                    col=2
                )

                # Line
                fig.add_scatter(
                    x=d_run['runtime_category'],
                    y=d_run[value_col],
                    mode='lines+markers',
                    name=t,
                    line=dict(color=colors[t], width=3),
                    legendgroup=t,
                    showlegend=False,
                    row=2,
                    col=1
                )

                # Scatter
                fig.add_scatter(
                    x=d_main['AMprime_Popularity'],
                    y=d_main[value_col],
                    mode='markers',
                    name=t,
                    marker=dict(color=colors[t], size=6, opacity=0.6),
                    legendgroup=t,
                    showlegend=False,
                    row=2,
                    col=2
                )

            fig.update_layout(
                template='plotly_dark',
                plot_bgcolor='#0F171E',
                paper_bgcolor='#0F171E',
                font=dict(color='#E6F1F5'),
                height=750,
                title=f"{title_name} Analysis Across Categories",
                title_x=0.5,
                legend=dict(
                    orientation="h",
                    y=1.05,
                    x=0.5,
                    xanchor="center"
                )
            )

            st.plotly_chart(fig, use_container_width=True)

        # SCORE MATRIX
        if bi_option == "AMprime Score Matrix":

            score_option = st.radio(
                "Select Analysis",
                [
                    "AMprime Score Category + Score",
                    "AMprime Score Genres & Production Countries"
                ],
                horizontal=True
            )

            if score_option == "AMprime Score Category + Score":

                st.markdown("""
                <div class="glass-card">
                    <h2>AMprime Score Analysis Across Categories</h2>
                </div>
                """, unsafe_allow_html=True)

                plot_multi_analysis(
                    Amazon_Prime,
                    'AMprime_Score',
                    'AMprime Score'
                )

                # ---------------------------------------------------
                # KEY INSIGHTS
                # ---------------------------------------------------

                st.markdown("""
                <div class="glass-card">

                <h3>Key Insights</h3>

                <ul>

                <li><b>Era Analysis</b>
                <ul>
                <li>AMprime Scores are relatively consistent across different eras.</li>
                <li>Movies maintain an average score around <b>5.9–6</b>.</li>
                <li>TV Shows consistently achieve higher scores between <b>7–7.5</b>.</li>
                </ul>
                </li>

                <li><b>Content Quality Analysis</b>
                <ul>
                <li><b>AMprime Rating Formula:</b>
                60% AMprime Score (IMDb + TMDb Score)
                + 40% AMprime Popularity
                (IMDb Votes + TMDb Popularity).</li>

                <li><b>Poor Quality:</b> Movies ≈ 6 | Shows ≈ 7</li>
                <li><b>Average Quality:</b> Movies ≈ 7 | Shows ≈ 8</li>
                <li><b>Good Quality:</b> Movies ≈ 7.9 | Shows ≈ 8.5</li>

                <li>Higher quality content strongly improves AMprime Score.</li>
                </ul>
                </li>

                <li><b>Runtime Category Analysis</b>
                <ul>
                <li><b>Long Runtime:</b> Movies ≈ 6.6 | Shows ≈ 7.1</li>
                <li><b>Short Runtime:</b> Movies ≈ 5.5 | Shows ≈ 7.1</li>
                <li><b>Average Runtime:</b> Movies ≈ 6 | Shows ≈ 7.3</li>
                </ul>
                </li>

                <li><b>Popularity Analysis</b>
                <ul>
                <li><b>Movies:</b> No strong correlation exists between popularity and AMprime Score.</li>
                <li>Most movies cluster around popularity 1–4 with scores between 5–8.</li>
                <li><b>TV Shows:</b> Most TV Shows consistently maintain high scores between 7–8.5.</li>
                <li>TV Shows appear significantly more stable and audience-favored.</li>
                </ul>
                </li>

                </ul>

                </div>
                """, unsafe_allow_html=True)

                # ---------------------------------------------------
                # BUSINESS INTERPRETATION
                # ---------------------------------------------------

                st.markdown("""
                <div class="glass-card">

                <h3>Business Interpretation</h3>

                <h4>Positive Business Impact</h4>

                <ul>

                <li><b>High-Quality TV Shows Drive Engagement</b>
                <ul>
                <li>TV Shows consistently achieve higher AMprime Scores.</li>
                <li>Investing in premium episodic content can improve retention and audience satisfaction.</li>
                </ul>
                </li>

                <li><b>Good Quality Content Builds Platform Credibility</b>
                <ul>
                <li>Content categorized as <b>Good</b> shows the highest and most stable scores.</li>
                <li>Prioritizing such content can strengthen brand perception and viewer trust.</li>
                </ul>
                </li>

                <li><b>Trending Content Improves Engagement</b>
                <ul>
                <li>Popularity shows a slight positive relationship with AMprime Score.</li>
                <li>Promoting trending content may increase watch time and platform interaction.</li>
                </ul>
                </li>

                </ul>

                <h4>Growth Challenges</h4>

                <ul>

                <li><b>High Variability in Movies</b>
                <ul>
                <li>Movie scores vary significantly from low to high.</li>
                <li>Inconsistent movie quality may negatively affect user experience and increase churn risk.</li>
                </ul>
                </li>

                <li><b>Poor Quality Content Impacts Brand Value</b>
                <ul>
                <li>Low-quality content receives significantly weaker scores.</li>
                <li>Excessive poor-performing titles may reduce platform credibility and user trust.</li>
                </ul>
                </li>

                </ul>

                </div>
                """, unsafe_allow_html=True)

            elif score_option == "AMprime Score Genres & Production Countries":

                st.markdown("""
                <div class="glass-card">
                    <h2>AMprime Score — Genres & Production Countries</h2>
                </div>
                """, unsafe_allow_html=True)

                fig = plot_metric(Amazon_Prime, 'AMprime_Score', 'AMprime Score')
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("""
                <div class="glass-card">

                <h3>Key Insights</h3>

                <ul>
                <li><b>Production countries:</b>
                <ul>
                <li>Movies: mean = 6.2 | Q75 = 6.8 to 7.2</li>
                <li>TV Shows: mean ≈ 7.3 | Q75 ≈ 7.8 to 8.0</li>
                </ul>
                </li>

                <li><b>Genres:</b>
                <ul>
                <li>Movies: mean ≈ 5.8 | Q75 ≈ 6.6</li>
                <li>TV Shows: mean ≈ 7.4 | Q75 ≈ 7.8 to 8.0</li>
                </ul>
                </li>
                </ul>

                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                <div class="glass-card">

                <h3>Business Interpretation</h3>

                <h4>Positive</h4>

                <ul>
                <li>TV Shows are more dominant than Movies with respect to gaining AMprime Score.</li>
                <li>Investing in top production countries combined with strong genres can drive customer engagement and ROI.</li>
                </ul>

                <h4>Negative</h4>

                <ul>
                <li>Most countries are not major producers of TV Shows, creating dependence on a smaller set of regions.</li>
                <li>Movies have a lower average AMprime Score (~6), which may increase churn risk.</li>
                <li>Low-performing countries and genres need strategic improvement to avoid underperformance.</li>
                </ul>

                </div>
                """, unsafe_allow_html=True)

        # POPULARITY MATRIX
        elif bi_option == "AMprime Popularity Matrix":

            popularity_option = st.radio(
                "Select Analysis",
                [
                    "AMprime Popularity Category + Score",
                    "AMprime Popularity Genres & Production Countries"
                ],
                horizontal=True
            )

            if popularity_option == "AMprime Popularity Category + Score":

                st.markdown("""
                <div class="glass-card">
                    <h2>AMprime Popularity Analysis Across Categories</h2>
                </div>
                """, unsafe_allow_html=True)

                plot_multi_analysis(Amazon_Prime, 'AMprime_Popularity', 'AMprime Popularity')

                st.markdown("""
                <div class="glass-card">

                <h3>Key Insights</h3>

                <ul>
                <li><b>Era Distribution:</b>
                <ul>
                <li>AMprime Popularity is almost equally distributed across eras: Movies ~2.4, TV Shows ~2.3</li>
                </ul>
                </li>

                <li><b>Content Quality Impact:</b>
                <ul>
                <li>Poor Quality: Movies ~2, Shows ~2</li>
                <li>Average Quality: Movies ~3.3, Shows ~2.8</li>
                <li>Good Quality: Movies ~5, Shows ~5.4</li>
                </ul>
                </li>

                <li><b>Runtime Category Analysis:</b>
                <ul>
                <li>Long Runtime: Movies ~2.8, Shows ~2.2</li>
                <li>Short Runtime: Movies ~1.8, Shows ~2.06</li>
                <li>Standard Runtime: Movies ~2.5, Shows ~2.4</li>
                </ul>
                </li>

                <li><b>Popularity Correlation:</b>
                <ul>
                <li>Strong direct proportional correlation between AMprime Popularity and TMDb Popularity for both types</li>
                <li>Movies show slightly higher peak popularity than TV Shows</li>
                <li>TV Shows maintain more stable popularity across eras</li>
                </ul>
                </li>
                </ul>

                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                <div class="glass-card">

                <h3>Business Interpretation</h3>

                <h4>Positive Impact</h4>

                <ul>
                <li>Movies are more popular than TV Shows overall with consistent growth across eras</li>
                <li>Good quality content significantly increases popularity (5x improvement from Poor to Good)</li>
                <li>Runtime optimization shows clear popularity benefits, especially for Movies</li>
                <li>Strong correlation with TMDb metrics validates our popularity measurement methodology</li>
                </ul>

                <h4>Challenges & Opportunities</h4>

                <ul>
                <li>Popularity has declined in Modern and Recent eras, suggesting shift towards TV Show preferences</li>
                <li>TV Shows show consistent but lower popularity, indicating need for better promotion strategies</li>
                <li>Quality differentiation is becoming evenly spread, reducing competitive advantage</li>
                <li>Focus on improving movie popularity while maintaining TV Show consistency</li>
                </ul>

                </div>
                """, unsafe_allow_html=True)

            elif popularity_option == "AMprime Popularity Genres & Production Countries":

                st.markdown("""
                <div class="glass-card">
                    <h2>AMprime Popularity — Genres & Production Countries</h2>
                </div>
                """, unsafe_allow_html=True)

                fig = plot_metric(Amazon_Prime, 'AMprime_Popularity', 'AMprime Popularity')
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("""
                <div class="glass-card">

                <h3>Key Insights</h3>

                <ul>
                <li><b>Production countries:</b>
                <ul>
                <li>Movies: mean = 2.6 | Q75 = 3.1 to 3.3</li>
                <li>TV Shows: mean = 2.4 to 2.6 | Q75 = 2.4 to 2.6</li>
                </ul>
                </li>

                <li><b>Genres:</b>
                <ul>
                <li>Movies: mean = 2.4 to 2.7 | Q75 = 3.0 to 3.3</li>
                <li>TV Shows: mean = 2.1 to 2.3 | Q75 = 2.6 to 2.8</li>
                </ul>
                </li>
                </ul>

                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                <div class="glass-card">

                <h3>Business Interpretation</h3>

                <h4>Positive</h4>

                <ul>
                <li>These production countries and genres are in the top 20, so the platform can focus on promoting and investing in them.</li>
                <li>Movies have slightly higher peak popularity while TV Shows are more stable.</li>
                <li>Combining top genres with top production countries can improve customer engagement and overall growth for both Movies and TV Shows.</li>
                </ul>

                <h4>Negative</h4>

                <ul>
                <li>Popularity growth is more driven by Movies than TV Shows, because only a few countries are producing most of the TV Shows.</li>
                <li>This creates a risk as the platform depends on limited regions for TV content.</li>
                <li>Genres and production countries with popularity below 2 should be avoided because they bring low engagement and low ROI.</li>
                </ul>

                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                <div class="glass-card">

                <h3>Key Insights</h3>

                <ul>
                <li><b>Production Countries:</b>
                <ul>
                <li>Movies: Mean ~2.6, Q75 ~3.1-3.3</li>
                <li>TV Shows: Mean ~2.4-2.6, Q75 ~2.4-2.6</li>
                <li>Movies show higher peak popularity with greater variability</li>
                <li>TV Shows maintain more consistent popularity across countries</li>
                </ul>
                </li>

                <li><b>Genres:</b>
                <ul>
                <li>Movies: Mean ~2.4-2.7, Q75 ~3.0-3.3</li>
                <li>TV Shows: Mean ~2.1-2.3, Q75 ~2.6-2.8</li>
                <li>Genre selection significantly impacts popularity for Movies</li>
                <li>Top genres show more consistent performance in TV Shows</li>
                </ul>
                </li>
                </ul>

                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                <div class="glass-card">

                <h3>Business Interpretation</h3>

                <h4>Positive Impact</h4>

                <ul>
                <li>Top 20 production countries and genres show strong popularity metrics suitable for platform focus</li>
                <li>Movies achieve higher peak popularity (3.3 vs 2.6 for TV Shows)</li>
                <li>Combining top genres with top production countries significantly improves customer engagement</li>
                <li>Strategic partnerships between countries can optimize content distribution</li>
                </ul>

                <h4>Risk Areas & Recommendations</h4>

                <ul>
                <li>Popularity growth is heavily driven by Movies; TV Shows depend on limited regions</li>
                <li>Platform risk from dependency on few TV content producers - diversification needed</li>
                <li>Genres and countries with popularity below 2.0 should be avoided - low engagement and ROI</li>
                <li>Recommend expanding TV Show production in emerging markets through partnerships</li>
                </ul>

                </div>
                """, unsafe_allow_html=True)

        # RATING MATRIX
        elif bi_option == "AMprime Rating Matrix":

            rating_option = st.radio(
                "Select Analysis",
                [
                    "AMprime Rating Category + Score",
                    "AMprime Rating Genres & Production Countries"
                ],
                horizontal=True
            )

            if rating_option == "AMprime Rating Category + Score":

                st.markdown("""
                <div class="glass-card">
                    <h2>AMprime Rating Analysis Across Categories</h2>
                </div>
                """, unsafe_allow_html=True)

                plot_multi_analysis(Amazon_Prime, 'AMprime_Rating', 'AMprime Rating')

                st.markdown("""
                <div class="glass-card">

                <h3>Key Insights</h3>

                <ul>
                <li><b>Era Analysis:</b>
                <ul>
                <li>Movies: AMprime Rating ~4.4 (±0.2) across eras</li>
                <li>TV Shows: AMprime Rating ~5.3 (±0.2) - consistently higher than Movies</li>
                <li>TV Shows show superior rating performance in every era</li>
                </ul>
                </li>

                <li><b>Content Quality Impact:</b>
                <ul>
                <li>Poor Quality: Movies 4.1-4.5, TV Shows 5.0-5.2</li>
                <li>Average Quality: Movies 5.0-5.4, TV Shows 5.5-6.2</li>
                <li>Good Quality: Movies 6.7-6.9, TV Shows 7.3+</li>
                <li>Quality categorization strongly differentiates Movies and TV Shows</li>
                </ul>
                </li>

                <li><b>Runtime Category Performance:</b>
                <ul>
                <li>Movies: Long runtime (~5.0) > Standard (~4.6) > Short (~4.0)</li>
                <li>TV Shows: Standard runtime (~5.3) > Long (~5.2) > Short (~5.1)</li>
                <li>Long runtime optimal for Movies; Standard runtime ideal for TV Shows</li>
                </ul>
                </li>

                <li><b>Popularity Correlation:</b>
                <ul>
                <li>Strong positive correlation between Popularity and Rating for both types</li>
                <li>Movies: Ratings exceed 6.0 when popularity exceeds 4.0</li>
                <li>TV Shows: Consistent ratings 4.0-6.0 across popularity range 1.5-3.5</li>
                <li>TV Shows show more predictable rating patterns</li>
                </ul>
                </li>
                </ul>

                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                <div class="glass-card">

                <h3>Business Interpretation</h3>

                <h4>Positive Opportunities</h4>

                <ul>
                <li>TV Shows significantly outperform Movies across all metrics (ratings ~20% higher)</li>
                <li>Despite lower content count, TV Shows show superior engagement and satisfaction</li>
                <li>Standard runtime (45-60 min) is ideal for TV Shows - aligns with binge-watching behavior</li>
                <li>Long runtime (150+ min) essential for Movies to achieve competitive ratings</li>
                <li>Strong investment case for TV Show expansion given consistent high ratings</li>
                </ul>

                <h4>Critical Challenges</h4>

                <ul>
                <li>Movie ratings declining era-by-era: Old era peak > Classic > Modern > Recent</li>
                <li>High proportion of Poor Content significantly impacts overall platform rating</li>
                <li>Movies need strategic intervention to reverse declining rating trend</li>
                <li>Quality content remains underrepresented compared to Poor quality content</li>
                <li>Recommendation: Shift focus towards TV Show investment for sustainable growth</li>
                </ul>

                </div>
                """, unsafe_allow_html=True)

            elif rating_option == "AMprime Rating Genres & Production Countries":

                st.markdown("""
                <div class="glass-card">
                    <h2>AMprime Rating — Genres & Production Countries</h2>
                </div>
                """, unsafe_allow_html=True)

                fig = plot_metric(Amazon_Prime, 'AMprime_Rating', 'AMprime Rating')
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("""
                <div class="glass-card">

                <h3>Key Insights</h3>

                <ul>
                <li><b>Production countries:</b> (+-0.2)
                <ul>
                <li>Movies: mean = 4.7 | Q75 = 5.3 | Upper = 6.8</li>
                <li>TV Shows: mean = 5.2 | Q75 = 5.7 | Upper = 6.7</li>
                </ul>
                </li>

                <li><b>Genres:</b> (+-0.2)
                <ul>
                <li>Movies: mean = 4.6 | Q75 = 5.1 | Upper = 7.0</li>
                <li>TV Shows: mean = 5.4 | Q75 = 5.8 | Upper = 7.2</li>
                </ul>
                </li>
                </ul>

                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                <div class="glass-card">

                <h3>Business Interpretation</h3>

                <h4>Positive</h4>

                <ul>
                <li>TV Shows are performing better than Movies across ratings and categories.</li>
                <li>Top production countries and genres can help increase customer engagement.</li>
                <li>Collaboration between developed and lower-income countries can reduce costs while maintaining quality.</li>
                </ul>

                <h4>Negative</h4>

                <ul>
                <li>The count of Movies is much higher, but their ratings are still low, which may hurt future growth.</li>
                <li>Avoid producing Movies in low-rated genres; these genres perform better as TV Shows.</li>
                <li>Low-rating countries should collaborate with developed markets to improve content quality and growth.</li>
                </ul>

                </div>
                """, unsafe_allow_html=True)

    # --------------------------------------------------------
    # MULTIVARIATE MATRIX
    # --------------------------------------------------------

    elif graph_menu == "Multivariate Matrix":

        st.header("Multivariate Matrix")

        multi_option = st.sidebar.radio(
            "Select Analysis Type",
            [
                "Heat Map",
                "Pair Plot"
            ]
        )

        if multi_option == "Heat Map":

            st.markdown("""
            <div class="glass-card">
                <h2>Correlation Heat Map</h2>
            </div>
            """, unsafe_allow_html=True)

            # Select numerical columns for correlation
            num_cols = ['imdb_score', 'tmdb_score', 'imdb_votes', 'tmdb_popularity', 'runtime', 'AMprime_Score', 'AMprime_Popularity', 'AMprime_Rating']

            # Calculate correlation matrix
            corr_matrix = Amazon_Prime[num_cols].corr()

            # Create heatmap
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='RdBu_r',
                title='Correlation Matrix of Numerical Variables'
            )

            fig.update_layout(
                template='plotly_dark',
                plot_bgcolor='#0f171e',
                paper_bgcolor='#0f171e',
                font=dict(color='white'),
                title_x=0.5,
                height=600
            )

            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            <div class="glass-card">

            <h3>Key Insights</h3>

            <ul>
            <li><b>Strong Positive Correlations:</b>
            <ul>
            <li>IMDb Score and TMDb Score (0.85) - Both rating systems are highly correlated</li>
            <li>IMDb Votes and TMDb Popularity (0.78) - Vote/popularity metrics align well</li>
            <li>AMprime Score and AMprime Rating (0.92) - Our composite metrics are strongly related</li>
            </ul>
            </li>

            <li><b>Moderate Correlations:</b>
            <ul>
            <li>AMprime Score and IMDb Score (0.81) - Score contributes significantly to our metric</li>
            <li>AMprime Popularity and TMDb Popularity (0.76) - Popularity metrics are related</li>
            </ul>
            </li>

            <li><b>Weak Correlations:</b>
            <ul>
            <li>Runtime has minimal correlation with scores/popularity</li>
            <li>IMDb Votes shows moderate correlation with scores</li>
            </ul>
            </li>
            </ul>

            </div>
            """, unsafe_allow_html=True)

        elif multi_option == "Pair Plot":

            st.markdown("""
            <div class="glass-card">
                <h2>Pair Plot Analysis</h2>
            </div>
            """, unsafe_allow_html=True)

            # Sample data for pair plot (to avoid performance issues with large dataset)
            sample_df = Amazon_Prime.sample(n=min(600, len(Amazon_Prime)), random_state=42)

            # Select key variables for pair plot
            pair_cols = ['imdb_score', 'tmdb_score', 'AMprime_Score', 'AMprime_Popularity', 'AMprime_Rating']

            # Create pair plot using plotly
            fig = px.scatter_matrix(
                sample_df,
                dimensions=pair_cols,
                color='type',
                color_discrete_map={'MOVIE': '#00A8E1', 'SHOW': '#FF9900'},
                title='Pair Plot of Key Metrics (Sample of 600 records)'
            )

            fig.update_layout(
                template='plotly_dark',
                plot_bgcolor='#0f171e',
                paper_bgcolor='#0f171e',
                font=dict(color='white'),
                title_x=0.5,
                height=800
            )

            # Update marker size
            fig.update_traces(diagonal_visible=False, marker=dict(size=3, opacity=0.6))

            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            <div class="glass-card">

            <h3>Key Insights</h3>

            <ul>
            <li><b>Distribution Patterns:</b>
            <ul>
            <li>TV Shows generally cluster at higher score ranges compared to Movies</li>
            <li>AMprime Score and Rating show strong linear relationships</li>
            <li>Popularity metrics show right-skewed distributions</li>
            </ul>
            </li>

            <li><b>Outlier Detection:</b>
            <ul>
            <li>Some Movies show exceptionally high popularity with moderate scores</li>
            <li>TV Shows maintain more consistent score distributions</li>
            </ul>
            </li>

            <li><b>Metric Relationships:</b>
            <ul>
            <li>IMDb and TMDb scores show strong positive correlation</li>
            <li>AMprime metrics provide balanced view of content quality</li>
            </ul>
            </li>
            </ul>

            </div>
            """, unsafe_allow_html=True)

# ============================================================
# INSIGHTS Q & A
# ============================================================

elif main_menu == "Insights Q & A":

    st.title("Insights Q & A - Solution to Business Objective")

    st.markdown("""
    <div class="glass-card">

    <h3>What do you suggest the client to achieve Business Objective ?</h3>

    <p>Our business objective is to enhance customer experience by providing high-quality Movies and TV Shows, which directly helps increase customer and viewer engagement on the platform. In this context, we discovered several key insights that are directly related to improving customer satisfaction and enhancing overall user experience.</p>

    <p>For stakeholders, we identified multiple business ideas that can create a positive impact on customer engagement and platform growth. We also discovered certain insights that may lead to negative growth and could potentially affect the overall user experience if not addressed properly.</p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">

    <h3>Factor 1:</h3>

    <p>First, the count of TV Show is almost 3 to 4 times less, but they are more popular in terms of IMDb score, TMDb score, votes, and popularity.</p>

    <ul>
    <li><b>Production countries:</b>
    <ul>
    <li>Top Countries in production: USA • UK • India • Canada • France • Germany • Japan • Italy • Australia • China</li>
    <li>Middle 10 Production Countries: Venezuela • Egypt • Turkey • Bolivia • Qatar • Singapore • Croatia • Lebanon • Bulgaria • Pakistan</li>
    <li>Bottom Production Countries: Cameroon • Fiji • Turks & Caicos Islands • Equatorial Guinea • Bangladesh • Libya • Antarctica • Paraguay • Dominican Republic • French Polynesia</li>
    </ul>
    </li>

    <li><b>Genres:</b>
    <ul>
    <li>Top Genres: Drama • Comedy • Thriller • Action • Romance • Crime • Horror • European • SciFi • Fantasy</li>
    <li>Middle Genres: Western • Family • Romance • Documentation • History</li>
    <li>Bottom Genres: Music • Sport • Reality • Documentation • Animation • War • History</li>
    </ul>
    </li>
    </ul>

    <h4>Business Growth Ideas</h4>

    <ul>
    <li><b>Top Segment:</b> All top countries are performing very well with top genres. There is no major need to focus heavily on this segment, as they are already generating outstanding customer engagement and ROI.</li>
    <li><b>Middle Segment:</b> These countries have strong future growth potential. A smart strategy would be to collaborate top production countries with these regions through smaller investments and regional partnerships. This can help the platform easily capture these markets, improve customer engagement, and achieve better ROI and market expansion.</li>
    <li><b>Bottom Segment:</b> Highly risky to invest in this segment with low chances to get ROI. But strategically, collaboration can work for future prospects. We can suggest that top segment producers make their TV Shows and Movies in these regions and give supporting roles to local actors, which builds audience from these regions and can still generate cost-effective ROI.</li>
    </ul>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">

    <h3>Factor 2:</h3>

    <ul>
    <li>Avoid producing Movies in bottom genres, as this can be highly risky and may lead to significant negative growth. Some middle-category genres may also perform poorly in Movies.</li>
    <li>Middle and bottom genres are not ideal for Movies, but they are gaining better scores and popularity in TV Shows. Hence, it would be a smart strategy to focus these genres on TV Shows, web series, and mini-series.</li>
    <li>This balanced approach can help normalize the scores and popularity of these genres while improving customer engagement and platform growth.</li>
    </ul>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">

    <h3>Factor 3:</h3>

    <p><b>Runtime Factor:</b></p>
    <ul>
    <li><b>Movies:</b> Standard to Long runtime is ideal for ratings, score, and popularity (above 150 minutes).</li>
    <li><b>TV Shows:</b> Standard runtime is ideal for gaining significant ratings and score (45-60 minutes).</li>
    </ul>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">

    <h3>Factor 4:</h3>

    <ul>
    <li>TV Shows have a lower content count, but they show significantly higher overall scores, popularity, and rating performance.</li>
    <li>Poor-quality content has a high impact on the overall rating system.</li>
    <li>The overall scoring system is more favorable for TV Show growth.</li>
    <li>The overall popularity system is more favorable for Movie growth.</li>
    <li>Considering all factors together, TV Shows show stronger overall growth potential.</li>
    </ul>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">

    <h3>Factor 5:</h3>

    <p>Users get more feasible and reliable recommendations with higher content quality, which can improve overall user experience.</p>

    <ul>
    <li><b>AMprime Score:</b> Mean of (IMDb Score + TMDb Score). Used to provide more trustworthy and high-quality content recommendations by combining the scoring systems of both IMDb and TMDb platforms.</li>
    <li><b>AMprime Popularity:</b> Mean of (IMDb Votes + TMDb Popularity). Used to provide more reliable and audience-based popularity measurement.</li>
    <li><b>AMprime Rating:</b> A combined metric created using 60% AMprime Score and 40% AMprime Popularity.</li>
    <li><b>Recommendation:</b> Balance quality and popularity so users receive more feasible and reliable recommendations, improving overall platform experience.</li>
    </ul>

    </div>
    """, unsafe_allow_html=True)

# ============================================================
# CONCLUSION
# ============================================================

elif main_menu == "Conclusion":

    st.title("Conclusion")

    st.markdown("""
    <div class="glass-card">

    <h3>Project Summary & Key Findings</h3>

    <p>The objective of this project was to improve customer experience and viewer engagement on the OTT platform through data-driven analysis of Movies and TV Shows.</p>

    <p>The analysis shows that TV Shows have stronger overall growth potential, with better ratings, scores, engagement, and popularity stability compared to Movies. Top production countries and genres are already generating high ROI and customer engagement, while middle-segment countries provide strong future growth opportunities through regional partnerships and strategic investments.</p>

    <p>Also found that many middle and bottom genres perform better in TV Shows, web series, and mini-series rather than Movies. Therefore, focusing these genres on episodic content would be a smarter business strategy.</p>

    <p>Overall, the insights from this project can help OTT platforms improve customer satisfaction, engagement, content strategy, global reach, and long-term business growth.</p>

    </div>
    """, unsafe_allow_html=True)
