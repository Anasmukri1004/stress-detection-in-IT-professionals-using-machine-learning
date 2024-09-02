import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from flask import *
import pickle

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/viewdata', methods=["GET", "POST"])
def viewdata():
    dataset = pd.read_csv('stress_detection_IT_professionals_dataset.csv')
    dataset.to_html()
    print(dataset)
    print(dataset.head(2))
    print(dataset.columns)
    return render_template("viewdata.html", columns=dataset.columns.values, rows=dataset.values.tolist())

@app.route('/preprocess', methods=['POST', 'GET'])
def preprocess():
    global x, y, x_train, x_test, y_train, y_test, df
    if request.method == "POST":
        size = int(request.form['split'])
        size = size / 100
        df = pd.read_csv('stress_detection_IT_professionals_dataset.csv')
        # Handle missing values
        df.dropna(inplace=True)  # Drop rows with missing values
        
        ##splitting
        x = df.drop('Stress_Level', axis=1)
        y = df['Stress_Level']

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        # describes info about train and test set
        print("Number transactions X_train dataset: ", x_train.shape)
        print("Number transactions y_train dataset: ", y_train.shape)
        print("Number transactions X_test dataset: ", x_test.shape)
        print("Number transactions y_test dataset: ", y_test.shape)

        print(x_train, x_test)

        return render_template('preprocess.html', msg='Data Preprocessed and It Splits Successfully')
    return render_template('preprocess.html')

@app.route('/model', methods=["POST", "GET"])
def model():
    if request.method == "POST":
        global x_train, x_test, y_train, y_test
        s = int(request.form['algo'])
        if s == 0:
            return render_template('model.html', msg="Choose an algorithm")

        elif s == 1:
            rf = RandomForestRegressor()
            rf.fit(x_train, y_train)
            y_pred = rf.predict(x_test)
            ac_rf = r2_score(y_test, y_pred)
            ac_rf = ac_rf * 100
            msg = "The r2_score obtained by RandomForestRegressor is " + str(ac_rf) + str('%')
            # Save the model
            with open('rf_model.pkl', 'wb') as f:
                pickle.dump(rf, f)
            return render_template("model.html", msg=msg)
        elif s == 2:
            ad = AdaBoostRegressor()
            ad.fit(x_train, y_train)
            y_pred = ad.predict(x_test)
            ac_ad = r2_score(y_test, y_pred)
            ac_ad = ac_ad * 100
            msg = "The r2_score obtained by AdaBoostRegressor " + str(ac_ad) + str('%')
            # Save the model
            with open('ad_model.pkl', 'wb') as f:
                pickle.dump(ad, f)
            return render_template("model.html", msg=msg)
        elif s == 3:
            ex = ExtraTreeRegressor()
            ex.fit(x_train, y_train)
            y_pred = ex.predict(x_test)
            ac_dt = r2_score(y_test, y_pred)
            ac_dt = ac_dt * 100
            msg = "The r2_score obtained by ExtraTreeRegressor is " + str(ac_dt) + str('%')
            # Save the model
            with open('ex_model.pkl', 'wb') as f:
                pickle.dump(ex, f)
            return render_template("model.html", msg=msg)
        elif s == 4:
            # Preprocess the data for CNN
            scaler = StandardScaler()
            scaler.fit(x_train, y_train)
            x_train_scaled = scaler.fit_transform(x_train)
            x_test_scaled = scaler.transform(x_test)

            # Reshape the data for CNN
            x_train_scaled = x_train_scaled[..., np.newaxis]
            x_test_scaled = x_test_scaled[..., np.newaxis]

            # Define the CNN model
            model = Sequential([
                Conv1D(64, kernel_size=3, activation='relu', input_shape=(x_train_scaled.shape[1], 1)),
                MaxPooling1D(pool_size=2),
                Flatten(),
                Dense(128, activation='relu'),
                Dropout(0.5),
                Dense(1, activation='linear')
            ])

            # Compile the model
            model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

            # Train the model
            model.fit(x_train_scaled, y_train, epochs=20, batch_size=32, validation_split=0.2)

            # Evaluate the model
            mse = model.evaluate(x_test_scaled, y_test)[1]
            r2 = mse -  1 / np.var(y_test)
            r2 = r2 * 100
            mae = mean_absolute_error(y_test, model.predict(x_test_scaled))
            rmse = np.sqrt(mse)
            msg = f"The r2_score obtained by CNN is {r2:.2f}%"
            return render_template("model.html", msg=msg)
            # Save the model
            model.save('cnn_model.h5')
    return render_template('model.html')

    return render_template("model.html")

# Route for prediction and stress reduction suggestion
@app.route('/prediction', methods=["POST", "GET"])
def prediction():
    if request.method == "POST":
        # Retrieve input values from the form
        f1 = request.form.get('Heart_Rate', type=float)
        f2 = request.form.get('Skin_Conductivity', type=float)
        f3 = request.form.get('Hours_Worked', type=float)
        f4 = request.form.get('Emails_Sent', type=float)
        f5 = request.form.get('Meetings_Attended', type=float)

        # Check if any of the inputs are None (indicating invalid input)
        if None in [f1, f2, f3, f4, f5]:
            return render_template('prediction.html', msg="Invalid input. Please enter valid numerical values.")

        # Prepare input for prediction
        lee = np.array([[f1, f2, f3, f4, f5]])

        # Load and train your RandomForestRegressor model
        model = RandomForestRegressor()
        model.fit(x_train, y_train)

        # Make prediction
        result = model.predict(lee)[0]

        # Determine suggestion based on stress level
        if result <= 20:
            suggestion = "Take a 10-minute walk outside, practice deep breathing exercises, or listen to calming music."
        elif result <= 40:
            suggestion = "Take a short break to stretch, practice progressive muscle relaxation, or try a quick mindfulness exercise."
        elif result <= 60:
            suggestion = "Take breaks and ask for help to improve your environment."
        elif result <= 75:
            suggestion = "Get enough sleep, organize your workspace, prioritize your most challenging tasks, and consider mindfulness practices."
        else:
            suggestion = "Reduce your caffeine intake, seek counseling, and explore stress management workshops."
 
        # Prepare message to display
        msg = f"The stress level of IT Professionals is {result:.2f}% - {suggestion}"
        return render_template('prediction.html', msg=msg)

    # Render the initial form if no POST request is made
    return render_template("prediction.html")

if __name__ == '__main__':
    app.run(debug=True)

if __name__ == "__main__":
    app.run(debug=True)
