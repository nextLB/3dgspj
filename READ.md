# 三维重构 -- 分块重建系统

conda create -n vasgaussian_system python=3.11


conda activate vasgaussian_system

pip install django 


之后来到vastgaussian文件夹下执行:

    pip  install -r requirements.txt

而后来到系统文件夹下:

    python manage.py makemigrations accounts projects                                                                                                    

    python manage.py migrate

    python manage.py seed_data 

    python manage.py runserver



