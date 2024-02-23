from pathlib import Path
import pandas as pd

def load_fraud_data(file_path: Path = Path("./data/fraud_detection_data.feather")) -> pd.DataFrame:
    return pd.read_feather(file_path)

def extract_node_data(df_fraud_data: pd.DataFrame) -> pd.DataFrame:
    # Combine all types of nodes into one table
    df_nodes = pd.concat([
        # One type of nodes are all the customers
        (
            df_fraud_data[["custom_person_hash"]]
            .drop_duplicates()
            .rename(columns={"custom_person_hash": "node_name"})
            .assign(node_type = "customer")
        ),
        # The other type of nodes is all the policies, contact details and vehicles
        (
            df_fraud_data[["link_value", "link_type"]]
            .drop_duplicates()
            .rename(columns={
                "link_value": "node_name",
                "link_type": "node_type",
            })
        )
    ]).reset_index(drop=True).reset_index(names=["node_id"])

    # We need to map the node types to integers so we can use them in the graph
    node_type_mapping = {category: i for i, category in enumerate(df_nodes["node_type"].unique())}
    df_nodes['node_type_int'] = df_nodes["node_type"].map(node_type_mapping)

    return df_nodes

def extract_edge_data(df_fraud_data: pd.DataFrame, df_nodes: pd.DataFrame) -> pd.DataFrame:
    # Get the node indices for the customer nodes
    df_customer_node_ids = (
        df_nodes[df_nodes["node_type"] == "customer"]
        .drop(columns="node_type")
        .rename(columns={
            "node_name": "custom_person_hash",
            "node_id": "node_from",
        })
    )
    # Get the node indices for all the policies, contact details and vehicles nodes
    df_other_node_ids = (
        df_nodes[df_nodes["node_type"] != "customer"]
        .drop(columns="node_type")
        .rename(columns={
            "node_name": "link_value",
            "node_id": "node_to",
        })
    )

    # Merge the node indices to the full list of all edges and keep only the indices and the link type
    df_edges = (
        df_fraud_data
        .merge(df_customer_node_ids, how="left", on=["custom_person_hash"])
        .merge(df_other_node_ids, how="left", on=["link_value"])
        [["node_from", "node_to", "link_type"]]
        .dropna()
        .drop_duplicates()
    )

    # We need to map the link types to integers so we can use them in the graph
    link_type_mapping = {category: i for i, category in enumerate(df_edges["link_type"].unique())}
    df_edges['link_type_int'] = df_edges["link_type"].map(link_type_mapping)

    return df_edges

def keep_only_customers_with_multi_edge_links(df_fraud_data: pd.DataFrame, relevant_link_types: list[str]) -> pd.DataFrame:
    # Keep only customers with multiple links
    customers_to_keep= []
    # Loop over the relevant link types
    for link_type in relevant_link_types:
        df_link_type = df_fraud_data[df_fraud_data["link_type"] == link_type]
        multiple_links = df_link_type.groupby('link_value')['custom_person_hash'].transform('nunique') > 1
        customers_to_keep += list(df_link_type[multiple_links]['custom_person_hash'].unique())

    # Make unique list of customers to keep
    customers_to_keep = list(set(customers_to_keep))

    return df_fraud_data[df_fraud_data["custom_person_hash"].isin(customers_to_keep)]
