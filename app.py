"""
CareerAI Web — Flask Backend
Genera el modelo automáticamente si no existe.
Deploy en Render: conecta tu repo de GitHub y listo.
"""
from flask import Flask, render_template, request, jsonify
import joblib, numpy as np, os

app = Flask(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))

FEATURES = [
    'logica_matematica', 'creatividad',       'habilidad_social',
    'interes_ciencias',  'interes_arte',
    'interes_tecnologia','liderazgo_organizacion','interes_humanidades',
]

def generar_modelo_si_no_existe():
    """Genera y guarda el modelo si los .pkl no están presentes."""
    model_path = os.path.join(BASE, 'careerai_model.pkl')
    if os.path.exists(model_path):
        return  # Ya existe, no hacer nada

    print("Generando modelo por primera vez...")
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    import warnings; warnings.filterwarnings('ignore')

    np.random.seed(42)
    CARRERAS = {
        'Ingenieria en Sistemas':             {'logica':9.5,'creatividad':8,  'social':3,  'ciencias':7,  'arte':2,  'tec':9.8,'lid':5,  'hum':2},
        'Medicina':                            {'logica':8,  'creatividad':5,  'social':9.5,'ciencias':9.5,'arte':4,  'tec':4,  'lid':7,  'hum':6},
        'Arquitectura':                        {'logica':6,  'creatividad':9.5,'social':5,  'ciencias':6,  'arte':9,  'tec':6,  'lid':5,  'hum':5},
        'Administracion de Empresas':          {'logica':6,  'creatividad':7,  'social':9,  'ciencias':4,  'arte':4,  'tec':5,  'lid':9.5,'hum':5},
        'Hoteleria Turismo y Gastronomia':     {'logica':4,  'creatividad':8.5,'social':9.5,'ciencias':3,  'arte':7,  'tec':4,  'lid':7,  'hum':6},
        'Contaduria Publica y Auditoria':      {'logica':7.5,'creatividad':4,  'social':6,  'ciencias':4,  'arte':2,  'tec':5,  'lid':6,  'hum':3},
        'Criminalistica y Ciencias Forenses':  {'logica':8,  'creatividad':6,  'social':8,  'ciencias':9,  'arte':3,  'tec':6,  'lid':6,  'hum':7},
        'Ingenieria Quimica':                  {'logica':7,  'creatividad':3,  'social':3,  'ciencias':9.8,'arte':2,  'tec':5,  'lid':4,  'hum':2},
        'Enfermeria':                          {'logica':5,  'creatividad':5,  'social':9.5,'ciencias':8.5,'arte':4,  'tec':3,  'lid':5,  'hum':7},
        'Derecho':                             {'logica':8,  'creatividad':7,  'social':9,  'ciencias':3,  'arte':5,  'tec':3,  'lid':8,  'hum':9.5},
        'Psicologia':                          {'logica':6,  'creatividad':8,  'social':9.5,'ciencias':6,  'arte':6,  'tec':3,  'lid':5,  'hum':9},
        'Diseno Grafico':                      {'logica':3,  'creatividad':9.8,'social':6,  'ciencias':2,  'arte':9.8,'tec':7,  'lid':4,  'hum':5},
        'Ingenieria Civil':                    {'logica':8.5,'creatividad':4,  'social':6,  'ciencias':8.5,'arte':3,  'tec':6,  'lid':6,  'hum':2},
        'Ingenieria Industrial':               {'logica':8.5,'creatividad':7,  'social':7,  'ciencias':7,  'arte':3,  'tec':7,  'lid':8,  'hum':3},
    }
    KEYS = ['logica','creatividad','social','ciencias','arte','tec','lid','hum']

    rows = []
    for carrera, p in CARRERAS.items():
        for _ in range(80):
            row = {FEATURES[i]: np.clip(np.random.normal(p[k], 0.9), 1, 10)
                   for i, k in enumerate(KEYS)}
            row['carrera'] = carrera
            rows.append(row)

    df   = pd.DataFrame(rows).sample(frac=1).reset_index(drop=True)
    X    = df[FEATURES].values
    le   = LabelEncoder();  y_enc = le.fit_transform(df['carrera'].values)
    sc   = StandardScaler(); X_sc  = sc.fit_transform(X)
    rf   = RandomForestClassifier(n_estimators=300, random_state=42)
    rf.fit(X_sc, y_enc)

    joblib.dump(rf, os.path.join(BASE, 'careerai_model.pkl'))
    joblib.dump(sc, os.path.join(BASE, 'careerai_scaler.pkl'))
    joblib.dump(le, os.path.join(BASE, 'careerai_labelencoder.pkl'))
    print("Modelo generado y guardado.")

# Generar modelo al arrancar si no existe
generar_modelo_si_no_existe()

def cargar_modelo():
    try:
        modelo = joblib.load(os.path.join(BASE, 'careerai_model.pkl'))
        scaler = joblib.load(os.path.join(BASE, 'careerai_scaler.pkl'))
        le     = joblib.load(os.path.join(BASE, 'careerai_labelencoder.pkl'))
        return modelo, scaler, le, True
    except FileNotFoundError:
        return None, None, None, False

modelo, scaler, le, modelo_ok = cargar_modelo()

@app.route('/')
def index():
    return render_template('index.html', modelo_ok=modelo_ok)

@app.route('/predecir', methods=['POST'])
def predecir():
    if not modelo_ok:
        return jsonify({'error': 'Modelo no disponible'}), 500
    data    = request.json
    valores = data.get('valores', {})
    perfil  = {f: float(np.mean(valores.get(f, [5.0]))) for f in FEATURES}
    entrada = np.array([[perfil[f] for f in FEATURES]])
    probs   = modelo.predict_proba(scaler.transform(entrada))[0]
    top_idx = np.argsort(probs)[::-1][:5]
    resultados = [
        {'carrera': le.classes_[i], 'pct': round(float(probs[i])*100, 1)}
        for i in top_idx
    ]
    return jsonify({'resultados': resultados, 'perfil': perfil})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))
    print(f"CareerAI Web corriendo en http://localhost:{port}")
    app.run(debug=False, host='127.0.0.1', port=port)
