import csv
import random

# Antal datapunkter
num_points = 5000000

# Antal dataset att skapa
num_datasets = 10

# Loopa igenom antal dataset
for j in range(1, num_datasets + 1):
    
    # Skapa en CSV-fil med unikt namn för varje iteration
    file_name = f"python_PRNG_dataset{j}.csv"
    with open(file_name, "w", newline='') as csvfile:
        fieldnames = ['Index', 'Value']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Skriv header
        writer.writeheader()

        # Loopa igenom antal datapunkter
        for i in range(1, num_points + 1):
            # Generera en randomiserad datapunkt (antingen 0 eller 1)
            value = random.randint(0, 1)
            
            # Skriv index och datapunkt till CSV-filen
            writer.writerow({'Index': i, 'Value': value})
