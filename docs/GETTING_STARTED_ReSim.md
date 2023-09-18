
# Getting Started

# ReSimAD dataset download

a. Register an account from OpenXLab website as follows.
```shell
https://openxlab.org.cn/home
```

b. Install the dependent libraries as follows:

* Install the openxlab dependent libraries.
  ```shell
    pip install openxlab
  ```
* Obtain the Access Key and Secret Key on the OpenXLab website by clicking the button of Account Security
* Login the OpenXLab using the Access Key and Secret Key
  ```shell
    openxlab login
  ```

c. Download the ReSimAD dataset by performing the following command:
```shell
openxlab dataset get --dataset-repo  Lonepic/ReSimAD
```

