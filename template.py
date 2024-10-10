import streamlit as st
import pandas as pd

if not st.checkbox("Show dataframe"):
    df = pd.DataFrame(range(0, 4))
    placeholder = st.empty()
    placeholder.dataframe(df)

    st.write("Chapter - 1")
    l = list(st.beta_columns(3))
    for i in range(len(l)):
        with l[i]:
            if st.checkbox("Subheading 1.{}".format(i+1)):
                st.write("More content (cols)")

    if st.checkbox("Chapter - 2"):
        l = list(st.beta_columns(3))
        
        with l[0]:
            if st.checkbox("Subheading 2.1"):
                st.write("This is subheading 2.1")
                st.image("https://render.githubusercontent.com/render/math?math=e^{i %2B\pi} =x%2B1")
                with l[1]:
                    if st.checkbox("Subheading 2.2"):
                        st.write("This is subheading 2.2")
                        with l[2]:
                            if st.checkbox("Subheading 2.3"):
                                st.write("This is subheading 2.3")

    if st.checkbox("Chapter - 3"):
        pass
        