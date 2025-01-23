import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

data = """
ADONIS                        |0.8621    |0.9259    |0.8929    |27        |0.9259  
AFRICAN GIANT SWALLOWTAIL     |0.9565    |0.9565    |0.9565    |23        |0.9565 
AMERICAN SNOOT                |0.9500    |0.8261    |0.8837    |23        |0.8261 
AN 88                         |1.0000    |1.0000    |1.0000    |26        |1.0000 
APPOLLO                       |1.0000    |0.8519    |0.9200    |27        |0.8519 
ATALA                         |0.9677    |1.0000    |0.9836    |30        |1.0000 
BANDED ORANGE HELICONIAN      |1.0000    |0.9667    |0.9831    |30        |0.9667 
BANDED PEACOCK                |0.8846    |0.9200    |0.9020    |25        |0.9200 
BECKERS WHITE                 |0.8800    |0.8800    |0.8800    |25        |0.8800 
BLACK HAIRSTREAK              |0.7857    |0.8462    |0.8148    |26        |0.8462 
BLUE MORPHO                   |1.0000    |0.8261    |0.9048    |23        |0.8261 
BLUE SPOTTED CROW             |0.9167    |0.8462    |0.8800    |26        |0.8462 
BROWN SIPROETA                |0.9062    |0.9667    |0.9355    |30        |0.9667 
CABBAGE WHITE                 |0.8966    |0.9630    |0.9286    |27        |0.9630 
CAIRNS BIRDWING               |0.9583    |0.9200    |0.9388    |25        |0.9200 
CHECQUERED SKIPPER            |0.9032    |0.9655    |0.9333    |29        |0.9655 
CHESTNUT                      |0.9630    |1.0000    |0.9811    |26        |1.0000 
CLEOPATRA                     |0.9200    |0.8214    |0.8679    |28        |0.8214 
CLODIUS PARNASSIAN            |0.8966    |0.9630    |0.9286    |27        |0.9630 
CLOUDED SULPHUR               |0.7037    |0.6786    |0.6909    |28        |0.6786 
COMMON BANDED AWL             |0.8636    |0.7037    |0.7755    |27        |0.7037 
COMMON WOOD-NYMPH             |0.8065    |0.9259    |0.8621    |27        |0.9259 
COPPER TAIL                   |0.6452    |0.6897    |0.6667    |29        |0.6897 
CRECENT                       |1.0000    |0.9667    |0.9831    |30        |0.9667 
CRIMSON PATCH                 |1.0000    |0.9545    |0.9767    |22        |0.9545 
DANAID EGGFLY                 |0.8889    |0.8276    |0.8571    |29        |0.8276 
EASTERN COMA                  |0.6944    |0.8929    |0.7812    |28        |0.8929 
EASTERN DAPPLE WHITE          |0.7407    |0.7143    |0.7273    |28        |0.7143 
EASTERN PINE ELFIN            |0.9032    |0.9655    |0.9333    |29        |0.9655 
ELBOWED PIERROT               |0.9200    |0.9200    |0.9200    |25        |0.9200 
GOLD BANDED                   |1.0000    |0.6818    |0.8108    |22        |0.6818 
GREAT EGGFLY                  |0.8261    |0.7917    |0.8085    |24        |0.7917 
GREAT JAY                     |0.9310    |0.9310    |0.9310    |29        |0.9310 
GREEN CELLED CATTLEHEART      |0.9524    |0.7407    |0.8333    |27        |0.7407 
GREY HAIRSTREAK               |0.8077    |0.8077    |0.8077    |26        |0.8077 
INDRA SWALLOW                 |0.7931    |0.9200    |0.8519    |25        |0.9200 
IPHICLUS SISTER               |1.0000    |0.8276    |0.9057    |29        |0.8276 
JULIA                         |0.9259    |1.0000    |0.9615    |25        |1.0000 
LARGE MARBLE                  |0.7000    |0.8400    |0.7636    |25        |0.8400 
MALACHITE                     |1.0000    |0.9545    |0.9767    |22        |0.9545 
MANGROVE SKIPPER              |0.8333    |0.9259    |0.8772    |27        |0.9259 
MESTRA                        |0.9167    |0.8462    |0.8800    |26        |0.8462 
METALMARK                     |0.8696    |0.8696    |0.8696    |23        |0.8696 
MILBERTS TORTOISESHELL        |0.8788    |1.0000    |0.9355    |29        |1.0000 
MONARCH                       |0.8889    |0.8889    |0.8889    |27        |0.8889 
MOURNING CLOAK                |0.9024    |0.9250    |0.9136    |40        |0.9250 
ORANGE OAKLEAF                |0.9000    |1.0000    |0.9474    |27        |1.0000 
ORANGE TIP                    |0.8710    |0.9310    |0.9000    |29        |0.9310 
ORCHARD SWALLOW               |0.9474    |0.7826    |0.8571    |23        |0.7826 
PAINTED LADY                  |0.9167    |0.9167    |0.9167    |24        |0.9167 
PAPER KITE                    |1.0000    |1.0000    |1.0000    |27        |1.0000 
PEACOCK                       |1.0000    |1.0000    |1.0000    |26        |1.0000 
PINE WHITE                    |0.9167    |0.8462    |0.8800    |26        |0.8462 
PIPEVINE SWALLOW              |0.7353    |0.9615    |0.8333    |26        |0.9615 
POPINJAY                      |0.9600    |0.9231    |0.9412    |26        |0.9231 
PURPLE HAIRSTREAK             |0.7273    |0.6667    |0.6957    |24        |0.6667 
PURPLISH COPPER               |0.7600    |0.6786    |0.7170    |28        |0.6786 
QUESTION MARK                 |0.7647    |0.5417    |0.6341    |24        |0.5417 
RED ADMIRAL                   |0.8696    |0.8000    |0.8333    |25        |0.8000 
RED CRACKER                   |1.0000    |0.9655    |0.9825    |29        |0.9655 
RED POSTMAN                   |0.9583    |0.8519    |0.9020    |27        |0.8519 
RED SPOTTED PURPLE            |1.0000    |0.9615    |0.9804    |26        |0.9615 
SCARCE SWALLOW                |0.9630    |0.8667    |0.9123    |30        |0.8667 
SILVER SPOT SKIPPER           |0.8621    |1.0000    |0.9259    |25        |1.0000 
SLEEPY ORANGE                 |0.8889    |0.9697    |0.9275    |33        |0.9697 
SOOTYWING                     |0.6154    |0.8889    |0.7273    |27        |0.8889 
SOUTHERN DOGFACE              |0.8261    |0.7037    |0.7600    |27        |0.7037 
STRAITED QUEEN                |1.0000    |0.9259    |0.9615    |27        |0.9259 
TROPICAL LEAFWING             |0.8846    |0.9200    |0.9020    |25        |0.9200 
TWO BARRED FLASHER            |1.0000    |1.0000    |1.0000    |23        |1.0000 
ULYSES                        |0.9615    |0.9615    |0.9615    |26        |0.9615 
VICEROY                       |0.8571    |0.9600    |0.9057    |25        |0.9600 
WOOD SATYR                    |0.9000    |0.8182    |0.8571    |22        |0.8182 
YELLOW SWALLOW TAIL           |0.8000    |0.8696    |0.8333    |23        |0.8696 
ZEBRA LONG WING               |1.0000    |1.0000    |1.0000    |23        |1.0000 
"""

df = pd.read_csv(StringIO(data), sep="|", header=None)
df.columns = ["Class Name", "Precision", "Recall", "F1-Score", "Support", "Accuracy"]
df["Class Name"] = df["Class Name"].str.strip()  
df["Accuracy"] = df["Accuracy"].astype(float)
df["F1-Score"] = df["F1-Score"].astype(float)

plt.figure(figsize=(14, 8))

plt.plot(df["Class Name"], df["Accuracy"], label="Accuracy", marker="o")
plt.plot(df["Class Name"], df["F1-Score"], label="F1-Score", marker="x")

plt.title("Class-wise Accuracy and F1-Score")
plt.xlabel("Class Name")
plt.ylabel("Metric Values")
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.show()

