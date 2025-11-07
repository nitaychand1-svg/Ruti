mermaid
graph TB
graph TB
    subgraph "ВХОДНЫЕ ДАННЫЕ"
        A[Технические индикаторы<br/>Объём/цена/время] --> B[Фильтр тяжёлых хвостов<br/>GARCH + PowerTransformer]
        B --> C[TimeSeriesSplit<br/>80% train / 20% val]
    end

    subgraph "УРОВЕНЬ 1: БАЗОВЫЕ МОДЕЛИ (64 шт)"
        direction TB
        C --> D1[RandomForest<br/>n_estimators=200]
        C --> D2[GradientBoosting<br/>learning_rate=0.05]
        C --> D3[ExtraTrees<br/>max_depth=10]
        C --> D4[AdaBoost<br/>n_estimators=100]
        C --> D5[SVM<br/>kernel='rbf']
        C --> D6[MLP<br/>hidden=(128,64)]
        C --> D7[LGBM<br/>n_estimators=300]
        C --> D8[CatBoost<br/>iterations=200]
        C --> D9[XGBoost<br/>max_depth=4]
        D1 --- D9
        D2 --- D9
        D3 --- D9
        
        D1 -.-> E[Модификация гиперпараметров<br/>Random diversity factor<br/>base_diversity=0.7]
        D2 -.-> E
        D3 -.-> E
        D4 -.-> E
        D5 -.-> E
        D6 -.-> E
        D7 -.-> E
        D8 -.-> E
        D9 -.-> E
    end

    subgraph "ФИЛЬТРАЦИЯ УРОВНЯ 1"
        D1 --> F1[Оценка точности<br/>Accuracy, AUC, Sharpe]
        D2 --> F1
        D3 --> F1
        D4 --> F1
        D5 --> F1
        D6 --> F1
        D7 --> F1
        D8 --> F1
        D9 --> F1
        
        F1 --> G[Pruning<br/>Удалить < 52% accuracy]
        G --> H[Пересчёт весов<br/>weight = 1/n_active]
    end
    
    subgraph "УРОВЕНЬ 2: BLENDER"
        direction TB
        H --> I[StackingClassifier<br/>TimeSeriesSplit(n_splits=3)]
        I --> J[Final Estimator<br/>LGBMClassifier<br/>n_estimators=200]
        J --> K[Статистики ансамбля<br/>mean, median, std<br/>25%, 75%, 5%, 95%]
        C --> K
        K --> L[Blender Input<br/>Features + Stats<br/>+ Top-10 models]
        L --> M[Blender Prediction<br/>predict_proba()]
    end
    
    subgraph "УРОВЕНЬ 3: META-РЕГУЛЯТОР"
        direction TB
        M --> N[Meta Features<br/>Blender prob + Base confidences<br/>+ Diversity score]
        
        subgraph "Market Regime Detection"
            O[Режим рынка<br/>low_volatility_bull<br/>low_volatility_bear<br/>high_volatility<br/>chaotic] --> P[Режим-специфичная калибровка]
        end
        
        N --> Q[CatBoost Meta-Model<br/>iterations=500<br/>depth=6<br/>eval_metric='AUC']
        O --> Q
        Q --> R[Raw Prediction]
        R --> P
        P --> S[Final Ensemble Prediction<br/>Calibrated probabilities]
    end
    
    subgraph "МОНИТОРИНГ & АДАПТАЦИЯ"
        direction LR
        T[Performance Tracker<br/>base/blender/meta<br/>correlation_matrix] --> U[Dynamic Weight Adaptation<br/>meta_adaptation_rate=0.1]
        M --> T
        Q --> T
        F1 --> T
        
        V[Prediction Cache<br/>{model_id: result}] --> W[Reuse for speed]
        D1 -.-> V
        D2 -.-> V
    end

    subgraph "РЕЖИМНАЯ ФИЛЬТРАЦИЯ"
        X[Market Regime Filter<br/>min_base_confidence=0.55<br/>max_correlation=0.85] --> Y[Active Models Selection<br/>top-10 if <5 active]
        F1 --> X
        X --> Y
        Y -.-> D1
        Y -.-> D2
        Y -.-> D8
    end
    
    subgraph "ДИВЕРСИФИКАЦИЯ"
        Z[Diversity Calculator<br/>1 - mean(|correlation|)] --> AA[Ensemble Diversity Score]
        D1 -.-> Z
        D2 -.-> Z
        D8 -.-> Z
        Z -.-> N
    end

    subgraph "ВЫХОД"
        S --> BB[Final Output<br/>ensemble_prediction<br/>blender_prediction<br/>base_predictions<br/>diversity<br/>active_models]
    end

    style A fill:#2c3e50,stroke:#333,stroke-width:2px,color:#fff
    style B fill:#34495e,stroke:#333,stroke-width:2px,color:#fff
    style C fill:#2c3e50,stroke:#333,stroke-width:2px,color:#fff
    
    style D1 fill:#27ae60,stroke:#333,stroke-width:2px,color:#fff
    style D2 fill:#27ae60,stroke:#333,stroke-width:2px,color:#fff
    style D8 fill:#e74c3c,stroke:#333,stroke-width:2px,color:#fff
    style D9 fill:#3498db,stroke:#333,stroke-width:2px,color:#fff
    
    style I fill:#f39c12,stroke:#333,stroke-width:3px,color:#fff
    style J fill:#f39c12,stroke:#333,stroke-width:2px,color:#fff
    style M fill:#f39c12,stroke:#333,stroke-width:2px,color:#fff
    
    style Q fill:#9b59b6,stroke:#333,stroke-width:3px,color:#fff
    style S fill:#9b59b6,stroke:#333,stroke-width:4px,color:#fff
    
    style V fill:#95a5a6,stroke:#333,stroke-width:2px,color:#fff
    style X fill:#e67e22,stroke:#333,stroke-width:2px,color:#fff
    
    style O fill:#d35400,stroke:#333,stroke-width:2px,color:#fff
    style P fill:#d35400,stroke:#333,stroke-width:2px,color:#fff
    
    style BB fill:#2c3e50,stroke:#333,stroke-width:4px,color:#fff
