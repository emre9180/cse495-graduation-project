import os
import shutil

# Paths
work_directory = r"C:\Users\emrey\OneDrive\Masaüstü\BitirmeFinal\yolov8\Test Set"
# target_directory = r"D:\INFRA\INFRA"
# target_directory = r"D:\results-fix-blue-33px\results-fix-blue-33px"
target_directory = r"D:\results-fix-rgb-33px\results-fix-rgb-33px"
# target_directory = r"D:\results-fix-red-33px\results-fix-red-33px"
thermal_directory = os.path.join(work_directory, 'thermal')
red_directory = os.path.join(work_directory, 'red')
blue_directory = os.path.join(work_directory, 'blue')
rgb_directory = os.path.join(work_directory, 'rgb')

# Create the thermal directory if it doesn't exist
if not os.path.exists(thermal_directory):
    os.makedirs(thermal_directory)

# Collect and rename files in work_directory
renamed_files = []
for filename in os.listdir(work_directory):
    if filename.endswith(".jpg"):
        # New file name
        # new_filename = filename.split('_jpg')[0] + '.jpg'
        renamed_files.append(filename)

        # # Rename file in the work directory
        # os.rename(os.path.join(work_directory, filename),
        #           os.path.join(work_directory, new_filename))

# Check and copy matching files from target_directory to thermal_directory
for filename in os.listdir(target_directory):
    if filename in renamed_files:
        # Copy file to the thermal directory
        shutil.copy2(os.path.join(target_directory, filename),
                     os.path.join(rgb_directory, filename))
