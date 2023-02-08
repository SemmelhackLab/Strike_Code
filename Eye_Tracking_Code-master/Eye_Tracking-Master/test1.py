import os
class StoreUserData(APIView):
    def post(self, request, *args, **kwargs):
        data = request.data
        fernet_obj = Fernet(key)
        try:
            lat = float(data.get('lat', None))
            long = float(data.get('long', None))
            search_item = data.get('search_item', None)
            date = data.get('date', None)
            time = data.get('time', None)
            email = data.get('email', None)
            phone = data.get('phone', None)
            res_details = data.get('res_name', None)

        except Exception as e:
            print(e)

        # Encrypting
        enc_phone = fernet_obj.encrypt(str(phone).encode()).decode()
        email = fernet_obj.encrypt(email.encode()).decode()
        phone = int(phone) + 8888888888
        phone = str(phone)

        filename = time + '.json'

        user_data = {}
        user_data['lat'] = lat
        user_data['long'] = long
        user_data['phone'] = enc_phone

        user_data['search_item'] = search_item.split(",")
        user_data['date'] = date
        user_data['time'] = time
        user_data['email'] = email
        user_data['res_details'] = res_details

        pwd = os.getcwd()
        # Changing current directorty to the user_data directory
        if os.path.isdir("/tmp/user_data"):
            os.chdir("/tmp/user_data/")
        else:
            os.mkdir("/tmp/user_data/")
            os.chdir("/tmp/user_data/")
        dirs = os.listdir()

        if phone in dirs:
            # get into the directory and create the date folder and then inside it create the file(time.json)
            os.chdir(phone + '/')
            if os.path.isdir(date):
                os.chdir(date + '/')
            else:
                os.mkdir(date)
                os.chdir(date + '/')
            print(os.getcwd())

        else:
            dir_name = phone + '/'
            os.mkdir(dir_name)
            os.chdir(dir_name + '/')
            sub_dir_name = date + '/'
            os.mkdir(sub_dir_name)
            os.chdir(sub_dir_name)
            print(os.getcwd())

        with open(filename, "w") as outfile:
            json.dump(user_data, outfile)
        os.chdir(pwd)
        return JsonResponse(user_data)