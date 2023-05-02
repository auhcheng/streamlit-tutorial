import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import py3Dmol
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import altair as alt
from io import BytesIO
import base64
# st.set_page_config(layout="wide")

####### BEGIN PART 1 #######
# text
st.title('Streamlit Tutorial')

st.write("We'll be doing some data science and machine learning with the Delaney solubility dataset.")

"You can write text like magic!"
"Markdown is *also* __supported__."

# TODO: try out some of the text elements from here: https://docs.streamlit.io/library/api-reference#text-elements

####### END PART 1 #########





















####### BEGIN PART 2 #######
# data, sliders, and images

@st.cache_data # what does this do? we'll worry about it later
def load_data():
    df = pd.read_csv('https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv')
    mols = [Chem.MolFromSmiles(smi) for smi in df['smiles']]
    return df, mols

df, mols = load_data()

"## Delaney solubility dataset"

st.write(df)

row_size = 4
num_to_display = st.number_input(
    label='Number of rows to display',
    min_value=1,
    max_value=5,
    value=1,
)

img = Draw.MolsToGridImage(mols[:num_to_display*row_size], molsPerRow=row_size)
st.image(img)

# notice how whenever any widget is changed, the entire script is re-run

# TODO: try out some of the other widgets from here: https://docs.streamlit.io/library/api-reference#display-interactive-widgets

# TODO: add a button that shuffles the images around
# https://docs.streamlit.io/library/api-reference/widgets/st.button

####### END PART 2 #########





















####### BEGIN PART 3 #######
# plotting, columns


"## Data exploration"
bins = st.slider('Number of bins', 5, 50, 20)
fig, ax = plt.subplots()
ax.hist(df['measured log solubility in mols per litre'], bins=bins)
ax.set_title('Histogram of solubility')
ax.set_xlabel('Measured log-solubility in mols/liter')
ax.set_ylabel('Number of compounds')
st.pyplot(fig) # show a plot


idx_to_display = st.slider(
    label='Index of mol to display',
    min_value=0,
    max_value=len(mols),
    value=0,
)

mol = mols[idx_to_display]

# create a 2-column layout
one, two = st.columns(2)

with one:
    # stuff in this block will go into the first column
    img = Draw.MolToImage(mol)
    st.image(img)

with two:
    # stuff in this block will go into the second column
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol)
    viewer = py3Dmol.view(width=400, height=400)
    viewer.addModel(Chem.MolToMolBlock(mol), 'mol')
    viewer.setStyle({'stick': {}})
    viewer.zoomTo()
    viewer.render()

    t = viewer.js()
    js = t.startjs + t.endjs

    st.components.v1.html(js, width=400, height=400)

st.table(df.iloc[idx_to_display].astype(str))

# TODO: add a button that will download the image of the molecule
# https://docs.streamlit.io/library/api-reference/widgets/st.download_button

# TODO: add a multiselect box that lets you pick which features to display
# https://docs.streamlit.io/library/api-reference/widgets/st.multiselect

####### END PART 3 #########





















####### BEGIN PART 4 #######
# machine learning, more plotting

"## Predicting solubility with Morgan fingerprints and BayesianRidge"

def get_morgan_fp(mol):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    fp_array = np.array(fp)
    return fp_array

@st.cache_data
def get_all_fp():
    return np.vstack([get_morgan_fp(mol) for mol in mols])
X = get_all_fp()

y = df['measured log solubility in mols per litre'].values

X_train, X_test, y_train, y_test, train_idxs, test_idxs = train_test_split(X, y, np.arange(len(mols)), shuffle=True, train_size=0.8, random_state=42)

@st.cache_data
def trained_model():
    return BayesianRidge().fit(X_train, y_train)
br = trained_model()

y_test_pred = br.predict(X_test)
st.write(f"MAE of BayesianRidge model: {mean_absolute_error(y_test, y_test_pred)}")

y_train_pred = br.predict(X_train)

df['split'] = 'test'
df.loc[train_idxs, 'split'] = 'train'
df.loc[train_idxs, 'predicted log solubility in mols per litre'] = y_train_pred
df.loc[test_idxs, 'predicted log solubility in mols per litre'] = y_test_pred


fig, ax = plt.subplots()
plt.scatter(y_train, y_train_pred, label="Train")
plt.scatter(y_test, y_test_pred, label="Test")
plt.legend()
plt.title("log-solubility in mols per litre")
plt.xlabel("Experimental")
plt.ylabel("Predicted")
st.pyplot(fig)


######### this part is for generating images to put in the altair chart #########
def image_formatter2(im):
    with BytesIO() as buffer:
        im.save(buffer, 'png')
        data = base64.encodebytes(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{data}"

@st.cache_data
def get_images():
    return [image_formatter2(Draw.MolToImage(mol)) for mol in mols]

df['image'] = get_images()
#################################################################################


c = alt.Chart(df).mark_circle(size=40).encode(
    y='predicted log solubility in mols per litre',
    x='measured log solubility in mols per litre',
    color=alt.Color('split', scale=alt.Scale(scheme='category10'), sort=['train', 'test']),
    tooltip=['smiles', 'measured log solubility in mols per litre', 'predicted log solubility in mols per litre', 'image'],
).interactive()

st.altair_chart(c, use_container_width=True)

# plotly, bokeh, pydeck, graphviz, vega-lite are all available: https://docs.streamlit.io/library/api-reference/charts

# TODO: try commenting out the @st cache decorators and see how long it takes to run

####### END PART 4 #########





















####### BEGIN PART 5 #######
# form

"## Predict solubility on your own SMILES string"

with st.form('input_form'):
    st.write('Input a SMILES string to predict log-solubility')
    """Examples:
- celecoxib: CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F
- MDMA: CC(CC1=CC2=C(C=C1)OCO2)NC
- aspirin: CC(=O)OC1=CC=CC=C1C(=O)O"""
    smi = st.text_input(
        label='SMILES',
        value='c1ccccc1',
    )
    submitted = st.form_submit_button('Predict')

    if submitted:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            st.write('Invalid SMILES')
        else:
            # create a 2-column layout
            one, two = st.columns(2)

            with one:
                # stuff in this block will go into the first column
                img = Draw.MolToImage(mol)
                st.image(img)

            with two:
                # stuff in this block will go into the second column
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol)
                AllChem.MMFFOptimizeMolecule(mol)
                viewer = py3Dmol.view(width=400, height=400)
                viewer.addModel(Chem.MolToMolBlock(mol), 'mol')
                viewer.setStyle({'stick': {}})
                viewer.zoomTo()
                viewer.render()

                t = viewer.js()
                js = t.startjs + t.endjs

                st.components.v1.html(js, width=400, height=400)
            
            fp = get_morgan_fp(mol)
            y_pred = br.predict(fp.reshape(1, -1)).item()
            st.write(f"Predicted log-solubility: {y_pred}")

            st.write("Molecules with closest measured log-solubility")
            distances = np.abs(y_pred - y)
            closest_idx = np.argsort(distances)[:5]
            closest_mols = [mols[idx] for idx in closest_idx]
            st.image(Draw.MolsToGridImage(closest_mols, molsPerRow=5))

# TODO: notice how the visualization code is the exact same as before in part 3
# refactor it into a function and call it twice

####### END PART 5 #########
