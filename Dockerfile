FROM gitlab-registry.cern.ch/allpix-squared/allpix-squared:latest

ENV PATH="/usr/local/bin:/usr/bin:${PATH}"

#RUN pip install --no-cache-dir \
    #numpy pandas scipy matplotlib seaborn \
    #scikit-learn xgboost "shap>=0.44" \
    #torch \
    #jupyterlab h5py uproot

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /project

CMD ["jupyter","lab","--ip=0.0.0.0","--allow-root","--no-browser"]
