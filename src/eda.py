import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# Φορτώνω το Dataset
df = pd.read_csv(r"C:\Users\odyba\Desktop\telco.csv")

# Ελέγχω αν λείπουν values σε κάθε column
missing_values = df.isnull().sum()

#Εμφανίζω columns που όντως τους λείπουν values
print(missing_values [missing_values > 0])

#Μετράω πόσοι πελάτες έφυγαν και πόσοι έμειναν
churn_counts = df['Churn'].value_counts()

print(churn_counts)

# θα εμφανίσω τις μετρήσεις σαν σχεδιάγραμμα με ράβδους
churn_counts.plot(kind='bar' , color=['blue','orange'])

#κώδικας για να έχει τίτλο
plt.title(' Customer Churn Count')

#κώδικας για ονοματίσω τον x άξονα
plt.xlabel('Churn')

#για να είναι κάθετα τα labels και να μήν κόβονται
plt.xticks(rotation=0)

# κώδικας για ονοματίσω τον y άξονα
plt.ylabel('Number of Customers')

plt.show()

#Δημιουργία  tenure bins δηλαδή ομαδοποιώ τους πελάτες με βάση το πόσο καιρό έιναι με την εταιρία
# Τα νούμερα που χρησιμοποιώ είναι οι μήνες σε διάρκεια
bins = [0, 12, 24, 36, 48, 60, 72]

#Δημιουργία λίστας των labels που ονοματίζουν και  πηγαίνουν μαζί με τα bins
labels = [ '0-12','12-24','24-36','36-48','48-60','60-72']

#Δημιουργία αντιστοίχησης τιμών με τα bins
#Δημιουργία νέας στήλης που αποθηκεύει τις κατηγορίες που ανήκουν οι πελάτες
df['TenureGroup'] = pd.cut(df['tenure'], bins=bins, labels=labels, right=False)

#Γκρουπάρω τα data με βάση την νέα στήλη που έφτιαξα με τα ναι και οχι μετράμε πόσοι έμειναν και πόσοι όχι
#Αναδιαμόρφοση πίνακα με unstack για σαφήνεια
tenure_churn = df.groupby('TenureGroup')['Churn'].value_counts().unstack()

#Σχεδίασμα διαγράμματος ωστε κάθε γραμμη να δείχνει και τις δύο ομάδες την μία πάνω στην άλλη
tenure_churn.plot(kind='bar', stacked=True)

plt.title('Churn by Tenure Group')

plt.xlabel('Tenure Group (months)')

plt.ylabel('Number of Customers')

plt.xticks(rotation=45)

plt.legend(title='Churn')

plt.show()

# Ομαδοποιήση των πελατων ανά τύπου σύμβασης
# Μέτρηση πόσοι έμειναν και πόσοι έχασαν πελάτες σε κάθε σύμβαση
# Καταγραφή των αποτελεσμάτων σε πίνακα.
print("Starting contract analysis...")
contract_churn = df.groupby('Contract')['Churn'].value_counts().unstack()
print(contract_churn)

#Αποτέλεσμα είναι οτι οι πελάτες με μηνιαία συμβόλαια είναι πολύ πιο πιθανό να αποχωρήσουν από τους πελάτες με μακροπρόθεσμα συμβόλαια.

#Εμφάνιση του σχεδιαγράμματος ανα μήνα ανα χρόνο και ανα δύο χρόνια
contract_churn.plot(kind='bar', stacked=True)
plt.title('Churn by Contract Type')
plt.xlabel('Contract Type')
plt.ylabel('Number of Customers')
plt.xticks(rotation=0)
plt.legend(title='Churn')
plt.show()

# Εμφάνιση των columns
print(df.columns)

# Δημιουργία κώδικα για να δούμε αν οι πελάτες που μηνιαία πληρώνουν περισσότερο φεύγουν ποιο συχνά
monthly_churn= df.groupby('Churn')['MonthlyCharges'].mean()
print(monthly_churn)
#Αποτέλεσμα είναι οτι οι πελάτες που έφυγαν είχαν σημαντικά υψηλότερες μηνιαίες χρεώσεις από τους πελάτες που παρέμειναν.
#Δημιουργία σχεδιαγράμματος
monthly_churn.plot(kind='bar')
plt.title('Average Monthly Charges by Churn Status')
plt.xlabel('Customer Status')
plt.ylabel(' Averagte Monthly Charges')
plt.xticks(rotation=0)
plt.show()

# Τσεκάρουμε το είδος των columns
print(df.dtypes)
# Εμφανίζω τα πρώτα 20 στοιχέια του totalcharges ωστε να ελεγξω οτι δεν έχουμε προβληματικά values
print(df['TotalCharges'].head(20))
print(df['TotalCharges'].tail(20))

#Ελέγχω για null values
print(df['TotalCharges'].isnull().sum())

# Εδώ μπορούμε να δούμε οτι το conent ενω είναι αριθμός αποθηυκεύεται ως text
print(type(df['TotalCharges'][0]))

#Διόρθωση
#df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
#print(df['TotalCharges'].dtype)

#Κάπου στο dataset υπάρχει κενό αντι για text που θέλαμε να το κάνουμε strng οπότε το αναγκάζουμε να μήν κρασάρει και
#μετατρέπουμε τα κε΄να σε NaN
df['TotalCharges'] = pd.to_numeric(
    df['TotalCharges'],
    errors='coerce'
)
print(df['TotalCharges'].dtype)
print(df['TotalCharges'].isnull().sum())

#Βλέπουμε οτι έχουμε 11 κενά τα οποία θα τα αφαιρέσουμε για να μην δημιουργηθεί πρόβλημα
# Τα μεταρέψαμε σε NaN και θα κάνουμε drop τα NaN
df = df.dropna()
print(df.shape)

#ελέγχουμε για missing values
print(df.isnull().sum())

#Διαχώριση της στήλης που θέλω να προβλέψω από τις πληροφορίες πελατών που θα χρησιμοποιήσω για να κάνω αυτήν την πρόβλεψη."
# το y είναι αυτό που θέλουμε να προβλέψουμε και το χ οτιδήποτε βοηθάει σε αυτήν την πρόβλεψη
y = df['Churn']

X = df.drop(columns=['Churn','customerID','Unnamed: 0' ])

print(X.shape)

print(y.shape)

# Επειδή υπ'αροχυν Values που είναι τεχτ και πρέπει να γίνουν αριθμοί πρέπει να κάνουμε encoding
print(X.dtypes)

# Μετατροπή όλων των πληροφορίων πελατών σε αριθμούς, ώστε να μπορούν να τις χρησιμοποιηθούν σε ένα μοντέλο μηχανικής μάθησης.
X_encoded = pd.get_dummies(X, drop_first=True)

print(X_encoded.shape)

# Χωριζμός του χ και του y σε μέρη εκπαίδευσης και δοκιμής
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y,test_size=0.2,random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#Εδώ θα κάνουμε το scaling
#Δημιουργία ενός προγράμματος κλιμάκωσης για να μάθεςι κανόνες κλιμάκωσης μόνο από τα δεδομένα εκπαίδευσης και, στη συνέχεια, εφαρμογή των κανόνων τόσο στις λειτουργίες εκπαίδευσης όσο και στις λειτουργίες δοκιμών.
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Δημιουργία Logistic Regression Model
model = LogisticRegression(max_iter=1000)

#Εκπαίδευση του model
model.fit(X_train_scaled, y_train)

#Βλέπουμε οτι πρέπει να κάνουμε scale καθώς τα values έχουν μεγάλη απόκλυση

#Τώρα ξεκινάνε τα predictions
y_pred = model.predict(X_test_scaled)
print(y_pred)

#Ελέγχουμε το πόσο σωστά έκανε τα predictions
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy) # Ηταν κατα 79,4% ακριβές το prediction

#Δημιουργία confusion matrix δηλαδή σύγκριση αποτελεσμάτων και εμφάνιση πίνακα
cm = confusion_matrix(y_test, y_pred)

print(cm)
#Αυτό είναι το αποτέλεσμα
#Actual customer	Model prediction	Count	Name
#Churned	Churned	199	True Positive
#Churned	Stayed	164	False Negative
#Stayed	Churned	111	False Positive
#Stayed	Stayed	860	True Negative

#Σύγκρηση των πραγματικών αποτελέσμάτων για τους πελάτες με τις προβλέψεις του μοντέλου και υπολογισμός της αξιοπιστίας του μοντέλου όταν προβλέπει ότι ένας πελάτης θα αποχωρήσει.
precision = precision_score(y_test, y_pred, pos_label='Churned')

print("Precision:", precision) #64% Ακρίβεια

#Υπολογισμός των πελατών που έφυγαν και εντοπίστηκαν με επιτυχία απο το μοντέλο
recall = recall_score(y_test, y_pred, pos_label='Churned')

print("Recall:", recall)

#Υπολογισμός συγκεκριμένου ποσοστού που λέει πόσο καλα το μοντέλο μπορέι σωστά να βρίσκει τους πελάτες που φεύγουν
f1 = f1_score(y_test, y_pred, pos_label='Churned')

print("F1 Score:", f1)


#Τα εμφανίζω όλα σε ένα report
print(classification_report(y_test, y_pred))
# Αποτέλεσμα Churned       0.64      0.55      0.59       363
#       Stayed       0.84      0.89      0.86       971
#
#     accuracy                           0.79      1334
#    macro avg       0.74      0.72      0.73      1334
# weighted avg       0.79      0.79      0.79      1334

#Ζητάω απο το μοντέλο να δ΄ωσεςι την πιθανότητα καθε test customer να ανήκει σε κάθε κλάση για τους πρώτους 5
y_pred = model.predict(X_test_scaled)

y_prob = model.predict_proba(X_test_scaled)
churn_prob = y_prob[:, 0]
print(y_prob[:5])
print(model.classes_)
print(churn_prob[:5])

#Υπολογισμός του πόσο καλα το μοντέλο διαχωρίζει τους πελάτες που φέυγουν απο τους πελάτες που μένουν
stayed_prob = y_prob[:, 1]

auc = roc_auc_score(y_test, stayed_prob)

print("AUC Score:", auc)

#Δοκιμή πολλών propability thresholds και υπολογισμό του ποσοστού false positive και true positive για τον σχεδιασμό της καμπύλης ROC
fpr, tpr, thresholds = roc_curve(y_test, stayed_prob, pos_label='Stayed')


#Φτιάχνω το grapgh της καμπύλης ROC
plt.figure(figsize=(8, 6))

plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')

plt.plot([0, 1], [0, 1], linestyle='--', label='Random Model')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.legend()

plt.show()

# Δημιουργία μικρότερου Threshold καθώς μας νοιάζει παραπάνω να φέυγουν λιγότεροι πελάτες
threshold = 0.30

y_pred_threshold = [
    'Churned' if prob >= threshold else 'Stayed'
    for prob in churn_prob
]

print(y_pred_threshold[:10])

# Εμφάνιση του καινούργιου matrix ωστε να συγκρίνουμε το 30 και το 50
cm_threshold = confusion_matrix(y_test, y_pred_threshold)

print(cm_threshold)


# Βλέπουμε οτι αυξήθηκε το ποσοστό του recall αυτό που θέλαμε δηλαδή
accuracy_threshold = accuracy_score(y_test, y_pred_threshold)

precision_threshold = precision_score(
    y_test,
    y_pred_threshold,
    pos_label='Churned'
)

recall_threshold = recall_score(
    y_test,
    y_pred_threshold,
    pos_label='Churned'
)

f1_threshold = f1_score(
    y_test,
    y_pred_threshold,
    pos_label='Churned'
)

print("New Accuracy:", accuracy_threshold)
print("New Precision:", precision_threshold)
print("New Recall:", recall_threshold)
print("New F1:", f1_threshold)
#Ανάλογα με το τι στόχο έχει η εταιρία μπορεί να συμφέρει η το 30 η το 50 threshold

#Καθώς το Logistic regression είναι linear και οι συνδιασμοί στην περίπτωση μας δεν είναι πάντα linear Θα δοκιμάσουμε
# άλλον αλγόριθμο τον Random Forest που είναι πιό ευέλικτος και θα τους συγκρίνουμε

#Δημιουργία οτυ μοντέλου με 100 decision trees
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Εκπαίδευση του μοντέλου
rf_model.fit(X_train_scaled, y_train)

# Δημιουργία προβλέψεων
rf_pred = rf_model.predict(X_test_scaled)

print(rf_pred[:10])

# Σύγκριση του μοντέλου με τα πραγματικά αποτελέσματα των πελατών
rf_accuracy = accuracy_score(y_test, rf_pred)

print("Random Forest Accuracy:", rf_accuracy)


# Υπολογισμός confusion matrix
rf_cm = confusion_matrix(y_test, rf_pred)

print(rf_cm)

# Υπολογισμός  precision , recall , f1 για καλύτερη σύγκριση
rf_precision = precision_score(
    y_test,
    rf_pred,
    pos_label='Churned'
)

rf_recall = recall_score(
    y_test,
    rf_pred,
    pos_label='Churned'
)

rf_f1 = f1_score(
    y_test,
    rf_pred,
    pos_label='Churned'
)

print("RF Precision:", rf_precision)
print("RF Recall:", rf_recall)
print("RF F1:", rf_f1)
#Βλέπουμε πως για αυτό το dataset το logistic regression έχει σε όλα τα επίπεδα καλύτερα στατιστικά απο το Random Forest


# Δημιουργία πίνακα με τα χαρακτηριστικά και το weight που του έχει αντιστοιχιστεί με το logistic regression
coefficients = pd.DataFrame({
    'Feature': X_encoded.columns,
    'Coefficient': model.coef_[0]
})

print(coefficients.head())


# Ταξινόμηση πίνακα απο τον μεγαλύτερο στον μικρότερο συντελεστή και αποθήκευση στις μεταβλητές συντελεστές
coefficients = coefficients.sort_values(
    by='Coefficient',
    ascending=False
)

print(coefficients.head(15))

#Για να δούμε γιατί φέυγουν παραπάνω
print(coefficients.tail(15))


# Δημιουργία σχεδιαγράμματος που δείχνει τα χαρακτηριστικά που αυξάνουν την διατήρηση πελατών
top_features = coefficients.head(10)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features['Feature'],
    top_features['Coefficient']
)

plt.title('Top 10 Positive Logistic Regression Coefficients')

# και το αντιθετο
bottom_features = coefficients.tail(10)

plt.figure(figsize=(10, 6))

plt.barh(
    bottom_features['Feature'],
    bottom_features['Coefficient']
)

plt.title('Top 10 Churn Drivers')

plt.xlabel('Coefficient Value')

plt.ylabel('Feature')

plt.show()


#Τελικά συμπεράσματα και τι θα πρότεινα βάση των αποτελεσμάτων
# Mακροπρόθεσμες συμβάσεις
# focus στους νεους πελάτες δηλαδη καλα πακέτα εκπτώσεις κτλπ
# Προώθηση πρόσθετων υπηρεσιών
# Διερεύνηση της ικανοποίησης πελατών οπτικών ινών καθώς είδαμε οτι είναι ενας απο τους λόγους που πελάτες φέυγουν
# Διερεύνηση Electronic Check πελατών για τον ίδιο λόγο



#Συμπεράσματα που έβγαλα απο αυτό το Project
#Το έργο ανέπτυξε με επιτυχία ένα σύστημα πρόβλεψης απώλειας πελατών χρησιμοποιώντας τεχνικές μηχανικής μάθησης.
# Το Logistic Regression πέτυχε ισχυρή προγνωστική απόδοση με βαθμολογία ROC-AUC 84,11% και ξεπέρασε την Random Forest
# σε αυτό το dataset. Η ανάλυση προσδιόρισε τη διάρκεια ζωής του πελάτη, τον τύπο σύμβασης, την τεχνική υποστήριξη
# και τις υπηρεσίες online ασφάλειας ως σημαντικούς παράγοντες διατήρησης, ενώ η υπηρεσία οπτικών ινών και οι
# ηλεκτρονικές πληρωμές με επιταγές συσχετίστηκαν με αυξημένο κίνδυνο απώλειας πελατών. Τα ευρήματα παρέχουν εφαρμόσιμες
# επιχειρηματικές συστάσεις που μπορούν να βοηθήσουν στη βελτίωση της διατήρησης των πελατών και στη μείωση της απώλειας εσόδων.

