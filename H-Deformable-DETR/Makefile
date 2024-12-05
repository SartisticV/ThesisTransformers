YML=docker/docker-compose.yml

include docker/.env
export

SHELL := /bin/bash

all:
	echo "	Usage:"
	echo "	make image 			# build image with docker-compose"
	echo "	make up 			# start container with gpu access"
	echo "	make exec 			# go into bash of container with gpu access"
	echo "	make bash 			# go into bash of container with gpu access"
	echo "	make down 			# stop all running services from docker-compose file"
	echo "	make rm 			# stop container with gpu access"
	echo "	make dlogs 			# show logs from inside the container"
	echo "	make mount 			# mount diskstation folders"
	echo "	make check_mount 	# print mounted diskstation folders"
	echo "	make jupyter_pw 	# create a jupyter pw"
	exho "	make jupyter 		# print the jupyter lab url"


image:
	docker-compose --env-file docker/.env -f $(YML) build

up:
	docker-compose --env-file docker/.env -f $(YML) up $(CONTAINER_NAME) -d

exec:
	docker exec -it $(CONTAINER_NAME) bash

bash:
	docker-compose --env-file docker/.env -f $(YML) up set_here_your_service_name -d
	docker exec -it v2119_model_compression bash

down:
	docker-compose --env-file docker/.env -f $(YML) down

rm:
	docker-compose --env-file docker/.env -f $(YML) rm -s

dlogs:
	docker logs -f $(CONTAINER_NAME)

# add more mounts if you want
mount:
	mkdir -p /data/${USER}/diskstation/${DISKSTATION_DIR_NAME} || true
	smbmount diskstationii1 $(DISKSTATION_DIR_NAME) /data/${USER}/diskstation/${DISKSTATION_DIR_NAME}

check_mount:
	mount | grep -w cifs

jupyter_pw:
	cd /common/${USER}/gitlab/python_basic/deep_learning/dl_machine_tools && ./jupyter_passwd.py ${PROJECT_DIR}

jupyter:
	echo "http://pc-11393:${PORT_JUPYTER}/lab"
